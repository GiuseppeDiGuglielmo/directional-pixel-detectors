їи'
ҐЖ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ђ
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleКйelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements#
handleКйelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint€€€€€€€€€
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
И"serve*2.7.02unknown8®™%
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@ *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

: *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
Т
time_distributed/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nametime_distributed/kernel
Л
+time_distributed/kernel/Read/ReadVariableOpReadVariableOptime_distributed/kernel*&
_output_shapes
:*
dtype0
В
time_distributed/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nametime_distributed/bias
{
)time_distributed/bias/Read/ReadVariableOpReadVariableOptime_distributed/bias*
_output_shapes
:*
dtype0
Ц
time_distributed_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nametime_distributed_1/kernel
П
-time_distributed_1/kernel/Read/ReadVariableOpReadVariableOptime_distributed_1/kernel*&
_output_shapes
: *
dtype0
Ж
time_distributed_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nametime_distributed_1/bias

+time_distributed_1/bias/Read/ReadVariableOpReadVariableOptime_distributed_1/bias*
_output_shapes
: *
dtype0
И
lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*&
shared_namelstm/lstm_cell/kernel
Б
)lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/kernel* 
_output_shapes
:
АА*
dtype0
Ы
lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*0
shared_name!lstm/lstm_cell/recurrent_kernel
Ф
3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel*
_output_shapes
:	@А*
dtype0

lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_namelstm/lstm_cell/bias
x
'lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
_output_shapes	
:А*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
В
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:@ *
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
: *
dtype0
Ж
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
†
Adam/time_distributed/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/time_distributed/kernel/m
Щ
2Adam/time_distributed/kernel/m/Read/ReadVariableOpReadVariableOpAdam/time_distributed/kernel/m*&
_output_shapes
:*
dtype0
Р
Adam/time_distributed/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/time_distributed/bias/m
Й
0Adam/time_distributed/bias/m/Read/ReadVariableOpReadVariableOpAdam/time_distributed/bias/m*
_output_shapes
:*
dtype0
§
 Adam/time_distributed_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/time_distributed_1/kernel/m
Э
4Adam/time_distributed_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/time_distributed_1/kernel/m*&
_output_shapes
: *
dtype0
Ф
Adam/time_distributed_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/time_distributed_1/bias/m
Н
2Adam/time_distributed_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/time_distributed_1/bias/m*
_output_shapes
: *
dtype0
Ц
Adam/lstm/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*-
shared_nameAdam/lstm/lstm_cell/kernel/m
П
0Adam/lstm/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/kernel/m* 
_output_shapes
:
АА*
dtype0
©
&Adam/lstm/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*7
shared_name(&Adam/lstm/lstm_cell/recurrent_kernel/m
Ґ
:Adam/lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp&Adam/lstm/lstm_cell/recurrent_kernel/m*
_output_shapes
:	@А*
dtype0
Н
Adam/lstm/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameAdam/lstm/lstm_cell/bias/m
Ж
.Adam/lstm/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/bias/m*
_output_shapes	
:А*
dtype0
В
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:@ *
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
: *
dtype0
Ж
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
†
Adam/time_distributed/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/time_distributed/kernel/v
Щ
2Adam/time_distributed/kernel/v/Read/ReadVariableOpReadVariableOpAdam/time_distributed/kernel/v*&
_output_shapes
:*
dtype0
Р
Adam/time_distributed/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/time_distributed/bias/v
Й
0Adam/time_distributed/bias/v/Read/ReadVariableOpReadVariableOpAdam/time_distributed/bias/v*
_output_shapes
:*
dtype0
§
 Adam/time_distributed_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/time_distributed_1/kernel/v
Э
4Adam/time_distributed_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/time_distributed_1/kernel/v*&
_output_shapes
: *
dtype0
Ф
Adam/time_distributed_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/time_distributed_1/bias/v
Н
2Adam/time_distributed_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/time_distributed_1/bias/v*
_output_shapes
: *
dtype0
Ц
Adam/lstm/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*-
shared_nameAdam/lstm/lstm_cell/kernel/v
П
0Adam/lstm/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/kernel/v* 
_output_shapes
:
АА*
dtype0
©
&Adam/lstm/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*7
shared_name(&Adam/lstm/lstm_cell/recurrent_kernel/v
Ґ
:Adam/lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp&Adam/lstm/lstm_cell/recurrent_kernel/v*
_output_shapes
:	@А*
dtype0
Н
Adam/lstm/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameAdam/lstm/lstm_cell/bias/v
Ж
.Adam/lstm/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
ёN
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЩN
valueПNBМN BЕN
у
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
	
signatures
#
_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
В
	layer
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
В
	layer
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
В
	layer
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
 	keras_api
В
	!layer
#"_self_saveable_object_factories
#	variables
$trainable_variables
%regularization_losses
&	keras_api
С
'cell
(
state_spec
#)_self_saveable_object_factories
*	variables
+trainable_variables
,regularization_losses
-	keras_api
Н

.kernel
/bias
#0_self_saveable_object_factories
1	variables
2trainable_variables
3regularization_losses
4	keras_api
Н

5kernel
6bias
#7_self_saveable_object_factories
8	variables
9trainable_variables
:regularization_losses
;	keras_api
Ь
<iter

=beta_1

>beta_2
	?decay
@learning_rate.mѓ/m∞5m±6m≤Am≥BmіCmµDmґEmЈFmЄGmє.vЇ/vї5vЉ6vљAvЊBvњCvјDvЅEv¬Fv√Gvƒ
 
 
N
A0
B1
C2
D3
E4
F5
G6
.7
/8
59
610
N
A0
B1
C2
D3
E4
F5
G6
.7
/8
59
610
 
≠
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
Н

Akernel
Bbias
#M_self_saveable_object_factories
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
 

A0
B1

A0
B1
 
≠
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
Н

Ckernel
Dbias
#W_self_saveable_object_factories
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
 

C0
D1

C0
D1
 
≠
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
w
#a_self_saveable_object_factories
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
 
 
 
 
≠
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
w
#k_self_saveable_object_factories
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
 
 
 
 
≠
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
#	variables
$trainable_variables
%regularization_losses
≥
u
state_size

Ekernel
Frecurrent_kernel
Gbias
#v_self_saveable_object_factories
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
 
 

E0
F1
G2

E0
F1
G2
 
Ї

{states
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
Аlayer_metrics
*	variables
+trainable_variables
,regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
 
≤
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
1	variables
2trainable_variables
3regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

50
61

50
61
 
≤
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
8	variables
9trainable_variables
:regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtime_distributed/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEtime_distributed/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtime_distributed_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtime_distributed_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUElstm/lstm_cell/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElstm/lstm_cell/recurrent_kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUElstm/lstm_cell/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE
 
1
0
1
2
3
4
5
6

Л0
М1
 
 
 

A0
B1

A0
B1
 
≤
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
 

0
 
 
 
 

C0
D1

C0
D1
 
≤
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
 

0
 
 
 
 
 
 
 
≤
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
 

0
 
 
 
 
 
 
 
≤
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
†layer_metrics
l	variables
mtrainable_variables
nregularization_losses
 

!0
 
 
 
 
 

E0
F1
G2

E0
F1
G2
 
≤
°non_trainable_variables
Ґlayers
£metrics
 §layer_regularization_losses
•layer_metrics
w	variables
xtrainable_variables
yregularization_losses
 
 

'0
 
 
 
 
 
 
 
 
 
 
 
 
 
8

¶total

Іcount
®	variables
©	keras_api
I

™total

Ђcount
ђ
_fn_kwargs
≠	variables
Ѓ	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

¶0
І1

®	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

™0
Ђ1

≠	variables
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/time_distributed/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/time_distributed/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/time_distributed_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/time_distributed_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/lstm_cell/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/lstm/lstm_cell/recurrent_kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/lstm/lstm_cell/bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/time_distributed/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/time_distributed/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/time_distributed_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/time_distributed_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/lstm_cell/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/lstm/lstm_cell/recurrent_kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/lstm/lstm_cell/bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
°
&serving_default_time_distributed_inputPlaceholder*3
_output_shapes!
:€€€€€€€€€*
dtype0*(
shape:€€€€€€€€€
≈
StatefulPartitionedCallStatefulPartitionedCall&serving_default_time_distributed_inputtime_distributed/kerneltime_distributed/biastime_distributed_1/kerneltime_distributed_1/biaslstm/lstm_cell/kernellstm/lstm_cell/biaslstm/lstm_cell/recurrent_kerneldense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*-
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *-
f(R&
$__inference_signature_wrapper_346980
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ћ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+time_distributed/kernel/Read/ReadVariableOp)time_distributed/bias/Read/ReadVariableOp-time_distributed_1/kernel/Read/ReadVariableOp+time_distributed_1/bias/Read/ReadVariableOp)lstm/lstm_cell/kernel/Read/ReadVariableOp3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp'lstm/lstm_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp2Adam/time_distributed/kernel/m/Read/ReadVariableOp0Adam/time_distributed/bias/m/Read/ReadVariableOp4Adam/time_distributed_1/kernel/m/Read/ReadVariableOp2Adam/time_distributed_1/bias/m/Read/ReadVariableOp0Adam/lstm/lstm_cell/kernel/m/Read/ReadVariableOp:Adam/lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp.Adam/lstm/lstm_cell/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp2Adam/time_distributed/kernel/v/Read/ReadVariableOp0Adam/time_distributed/bias/v/Read/ReadVariableOp4Adam/time_distributed_1/kernel/v/Read/ReadVariableOp2Adam/time_distributed_1/bias/v/Read/ReadVariableOp0Adam/lstm/lstm_cell/kernel/v/Read/ReadVariableOp:Adam/lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp.Adam/lstm/lstm_cell/bias/v/Read/ReadVariableOpConst*7
Tin0
.2,	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *(
f#R!
__inference__traced_save_349603
€	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetime_distributed/kerneltime_distributed/biastime_distributed_1/kerneltime_distributed_1/biaslstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biastotalcounttotal_1count_1Adam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/time_distributed/kernel/mAdam/time_distributed/bias/m Adam/time_distributed_1/kernel/mAdam/time_distributed_1/bias/mAdam/lstm/lstm_cell/kernel/m&Adam/lstm/lstm_cell/recurrent_kernel/mAdam/lstm/lstm_cell/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/time_distributed/kernel/vAdam/time_distributed/bias/v Adam/time_distributed_1/kernel/vAdam/time_distributed_1/bias/vAdam/lstm/lstm_cell/kernel/v&Adam/lstm/lstm_cell/recurrent_kernel/vAdam/lstm/lstm_cell/bias/v*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *+
f&R$
"__inference__traced_restore_349739Љд#
Ь
°
L__inference_time_distributed_layer_call_and_return_conditional_losses_347748

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identityИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€К
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0≤
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Т
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Ќ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:Р
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€Е
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&€€€€€€€€€€€€€€€€€€: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Т?
©
E__inference_lstm_cell_layer_call_and_return_conditional_losses_349347

inputs
states_0
states_11
split_readvariableop_resource:
АА.
split_1_readvariableop_resource:	А*
readvariableop_resource:	@А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґsplit/ReadVariableOpҐsplit_1/ReadVariableOpG
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ґ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitZ
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€@\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€@\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€@\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€@S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Ф
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@Z
mulMulstates_0ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@\
mul_1Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@\
mul_2Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@\
mul_3Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      л
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maske
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€@i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@W
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€@i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@_
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@V
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@K
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@c
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@X
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Z

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@ј
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€А:€€€€€€€€€@:€€€€€€€€€@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/1
µ
Ђ
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_347892

inputsA
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: 
identityИҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€О
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ґ
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 *
paddingVALID*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 l
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€   	          Й
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€	 m
IdentityIdentityReshape_1:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€	 Й
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
т
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_345459

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ∆
flatten/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_345425\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :АХ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:Р
	Reshape_1Reshape flatten/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Аh
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&€€€€€€€€€€€€€€€€€€ :d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
ЪЪ
 
@__inference_lstm_layer_call_and_return_conditional_losses_349128

inputs;
'lstm_cell_split_readvariableop_resource:
АА8
)lstm_cell_split_1_readvariableop_resource:	А4
!lstm_cell_readvariableop_resource:	@А
identityИҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€АD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_maskW
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Х
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@\
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @О
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@c
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:Њ
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ўЛ;e
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ƒ
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Г
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@З
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@^
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Т
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:√
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2‘щХg
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Н
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@^
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Т
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:√
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2њГЙg
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Н
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@^
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Т
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:√
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2фГцg
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Н
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0ј
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitА
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : З
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0≤
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitЖ
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@s
lstm_cell/mulMulzeros:output:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@w
lstm_cell/mul_1Mulzeros:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@w
lstm_cell/mul_2Mulzeros:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@w
lstm_cell/mul_3Mulzeros:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@{
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Э
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskГ
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@s
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@]
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@t
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@_
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Б
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : п
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_348969*
condR
while_cond_348968*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Д
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€А: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
пv
 
@__inference_lstm_layer_call_and_return_conditional_losses_348835

inputs;
'lstm_cell_split_readvariableop_resource:
АА8
)lstm_cell_split_1_readvariableop_resource:	А4
!lstm_cell_readvariableop_resource:	@А
identityИҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€АD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_maskW
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Х
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0ј
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitА
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : З
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0≤
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitЖ
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@t
lstm_cell/mulMulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@v
lstm_cell/mul_1Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@v
lstm_cell/mul_2Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@v
lstm_cell/mul_3Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@{
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Э
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskГ
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@s
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@]
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@t
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@_
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Б
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : п
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_348708*
condR
while_cond_348707*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Д
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€А: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ЫЪ
 
@__inference_lstm_layer_call_and_return_conditional_losses_346627

inputs;
'lstm_cell_split_readvariableop_resource:
АА8
)lstm_cell_split_1_readvariableop_resource:	А4
!lstm_cell_readvariableop_resource:	@А
identityИҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€АD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_maskW
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Х
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@\
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @О
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@c
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:њ
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2≈Вїe
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ƒ
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Г
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@З
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@^
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Т
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:√
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2≠У≠g
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Н
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@^
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Т
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:√
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2еƒ∆g
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Н
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@^
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Т
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:√
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2нБЗg
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Н
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0ј
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitА
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : З
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0≤
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitЖ
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@s
lstm_cell/mulMulzeros:output:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@w
lstm_cell/mul_1Mulzeros:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@w
lstm_cell/mul_2Mulzeros:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@w
lstm_cell/mul_3Mulzeros:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@{
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Э
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskГ
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@s
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@]
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@t
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@_
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Б
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : п
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_346468*
condR
while_cond_346467*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Д
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€А: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
П
¶
1__inference_time_distributed_layer_call_fn_347700

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_346734{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
µ
√
while_cond_348968
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_348968___redundant_placeholder04
0while_while_cond_348968___redundant_placeholder14
0while_while_cond_348968___redundant_placeholder24
0while_while_cond_348968___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
Ў
Џ
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_345287

inputs)
conv2d_1_345275: 
conv2d_1_345277: 
identityИҐ conv2d_1/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€З
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_1_345275conv2d_1_345277*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€	 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_345274\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : Ќ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:†
	Reshape_1Reshape)conv2d_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 i
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&€€€€€€€€€€€€€€€€€€: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
 
Ђ
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_347838

inputsA
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: 
identityИҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€О
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ґ
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 *
paddingVALID*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 \
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : Ќ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:Т
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 Й
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&€€€€€€€€€€€€€€€€€€: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ЁV
Љ
__inference__traced_save_349603
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_time_distributed_kernel_read_readvariableop4
0savev2_time_distributed_bias_read_readvariableop8
4savev2_time_distributed_1_kernel_read_readvariableop6
2savev2_time_distributed_1_bias_read_readvariableop4
0savev2_lstm_lstm_cell_kernel_read_readvariableop>
:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop2
.savev2_lstm_lstm_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop=
9savev2_adam_time_distributed_kernel_m_read_readvariableop;
7savev2_adam_time_distributed_bias_m_read_readvariableop?
;savev2_adam_time_distributed_1_kernel_m_read_readvariableop=
9savev2_adam_time_distributed_1_bias_m_read_readvariableop;
7savev2_adam_lstm_lstm_cell_kernel_m_read_readvariableopE
Asavev2_adam_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop9
5savev2_adam_lstm_lstm_cell_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop=
9savev2_adam_time_distributed_kernel_v_read_readvariableop;
7savev2_adam_time_distributed_bias_v_read_readvariableop?
;savev2_adam_time_distributed_1_kernel_v_read_readvariableop=
9savev2_adam_time_distributed_1_bias_v_read_readvariableop;
7savev2_adam_lstm_lstm_cell_kernel_v_read_readvariableopE
Asavev2_adam_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop9
5savev2_adam_lstm_lstm_cell_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Л
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*і
value™BІ+B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH√
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B В
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_time_distributed_kernel_read_readvariableop0savev2_time_distributed_bias_read_readvariableop4savev2_time_distributed_1_kernel_read_readvariableop2savev2_time_distributed_1_bias_read_readvariableop0savev2_lstm_lstm_cell_kernel_read_readvariableop:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop.savev2_lstm_lstm_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop9savev2_adam_time_distributed_kernel_m_read_readvariableop7savev2_adam_time_distributed_bias_m_read_readvariableop;savev2_adam_time_distributed_1_kernel_m_read_readvariableop9savev2_adam_time_distributed_1_bias_m_read_readvariableop7savev2_adam_lstm_lstm_cell_kernel_m_read_readvariableopAsavev2_adam_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop5savev2_adam_lstm_lstm_cell_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop9savev2_adam_time_distributed_kernel_v_read_readvariableop7savev2_adam_time_distributed_bias_v_read_readvariableop;savev2_adam_time_distributed_1_kernel_v_read_readvariableop9savev2_adam_time_distributed_1_bias_v_read_readvariableop7savev2_adam_lstm_lstm_cell_kernel_v_read_readvariableopAsavev2_adam_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop5savev2_adam_lstm_lstm_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*х
_input_shapesг
а: :@ : : :: : : : : ::: : :
АА:	@А:А: : : : :@ : : :::: : :
АА:	@А:А:@ : : :::: : :
АА:	@А:А: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :,
(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
АА:%!

_output_shapes
:	@А:!

_output_shapes	
:А:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
АА:%!

_output_shapes
:	@А:!

_output_shapes	
:А:$  

_output_shapes

:@ : !

_output_shapes
: :$" 

_output_shapes

: : #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
::,&(
&
_output_shapes
: : '

_output_shapes
: :&("
 
_output_shapes
:
АА:%)!

_output_shapes
:	@А:!*

_output_shapes	
:А:+

_output_shapes
: 
µ
√
while_cond_346106
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_346106___redundant_placeholder04
0while_while_cond_346106___redundant_placeholder14
0while_while_cond_346106___redundant_placeholder24
0while_while_cond_346106___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
В?
І
E__inference_lstm_cell_layer_call_and_return_conditional_losses_345572

inputs

states
states_11
split_readvariableop_resource:
АА.
split_1_readvariableop_resource:	А*
readvariableop_resource:	@А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґsplit/ReadVariableOpҐsplit_1/ReadVariableOpE
ones_like/ShapeShapestates*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ґ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitZ
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€@\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€@\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€@\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€@S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Ф
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@X
mulMulstatesones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Z
mul_1Mulstatesones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Z
mul_2Mulstatesones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Z
mul_3Mulstatesones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      л
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maske
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€@i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@W
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€@i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@_
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@V
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@K
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@c
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@X
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Z

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@ј
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€А:€€€€€€€€€@:€€€€€€€€€@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates
µ
√
while_cond_345585
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_345585___redundant_placeholder04
0while_while_cond_345585___redundant_placeholder14
0while_while_cond_345585___redundant_placeholder24
0while_while_cond_345585___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
ъ
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_345374

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 ў
max_pooling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_345365\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : Ќ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:Э
	Reshape_1Reshape&max_pooling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ o
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&€€€€€€€€€€€€€€€€€€	 :d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 
 
_user_specified_nameinputs
љ
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_348040

inputs
identityf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   w
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аd
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€      
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€А_
IdentityIdentityReshape_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ :[ W
3
_output_shapes!
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ш

т
A__inference_dense_layer_call_and_return_conditional_losses_349148

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ј
“
L__inference_time_distributed_layer_call_and_return_conditional_losses_345201

inputs'
conv2d_345189:
conv2d_345191:
identityИҐconv2d/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€€
conv2d/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_345189conv2d_345191*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_345188\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Ќ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:Ю
	Reshape_1Reshape'conv2d/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€g
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&€€€€€€€€€€€€€€€€€€: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
¶"
–
while_body_345852
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_345876_0:
АА'
while_lstm_cell_345878_0:	А+
while_lstm_cell_345880_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_345876:
АА%
while_lstm_cell_345878:	А)
while_lstm_cell_345880:	@АИҐ'while/lstm_cell/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€А*
element_dtype0©
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_345876_0while_lstm_cell_345878_0while_lstm_cell_345880_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_345793ў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Н
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Н
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@v

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"2
while_lstm_cell_345876while_lstm_cell_345876_0"2
while_lstm_cell_345878while_lstm_cell_345878_0"2
while_lstm_cell_345880while_lstm_cell_345880_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
К
O
3__inference_time_distributed_2_layer_call_fn_347897

inputs
identity”
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_345374u
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&€€€€€€€€€€€€€€€€€€	 :d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 
 
_user_specified_nameinputs
С
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_349222

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ъ
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_348003

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   w
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :АХ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:И
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Аh
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&€€€€€€€€€€€€€€€€€€ :d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
л
Ь
'__inference_conv2d_layer_call_fn_349176

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_345188w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ъ
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_348020

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   w
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :АХ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:И
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Аh
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&€€€€€€€€€€€€€€€€€€ :d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
С
µ
%__inference_lstm_layer_call_fn_348062
inputs_0
unknown:
АА
	unknown_0:	А
	unknown_1:	@А
identityИҐStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_345921o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А
"
_user_specified_name
inputs/0
±Р
В	
while_body_348969
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
АА@
1while_lstm_cell_split_1_readvariableop_resource_0:	А<
)while_lstm_cell_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
АА>
/while_lstm_cell_split_1_readvariableop_resource:	А:
'while_lstm_cell_readvariableop_resource:	@АИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€А*
element_dtype0b
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:d
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?І
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@b
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @†
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@o
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
: 
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ЩЋ]k
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?÷
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@П
while/lstm_cell/dropout/CastCast(while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Щ
while/lstm_cell/dropout/Mul_1Mulwhile/lstm_cell/dropout/Mul:z:0 while/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@d
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @§
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:ѕ
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2£эрm
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?№
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@У
while/lstm_cell/dropout_1/CastCast*while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Я
while/lstm_cell/dropout_1/Mul_1Mul!while/lstm_cell/dropout_1/Mul:z:0"while/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@d
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @§
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:ѕ
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2 ’€m
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?№
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@У
while/lstm_cell/dropout_2/CastCast*while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Я
while/lstm_cell/dropout_2/Mul_1Mul!while/lstm_cell/dropout_2/Mul:z:0"while/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@d
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @§
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:ѕ
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2“Уџm
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?№
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@У
while/lstm_cell/dropout_3/CastCast*while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Я
while/lstm_cell/dropout_3/Mul_1Mul!while/lstm_cell/dropout_3/Mul:z:0"while/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0“
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_split§
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Х
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0ƒ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitШ
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@Д
while/lstm_cell/mulMulwhile_placeholder_2!while/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@И
while/lstm_cell/mul_1Mulwhile_placeholder_2#while/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@И
while/lstm_cell/mul_2Mulwhile_placeholder_2#while/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@И
while/lstm_cell/mul_3Mulwhile_placeholder_2#while/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Й
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskХ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ф
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@В
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@i
while/lstm_cell/ReluReluwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@П
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@У
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@¬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@¶

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
пл
л
!__inference__wrapped_model_345163
time_distributed_input[
Asequential_time_distributed_conv2d_conv2d_readvariableop_resource:P
Bsequential_time_distributed_conv2d_biasadd_readvariableop_resource:_
Esequential_time_distributed_1_conv2d_1_conv2d_readvariableop_resource: T
Fsequential_time_distributed_1_conv2d_1_biasadd_readvariableop_resource: K
7sequential_lstm_lstm_cell_split_readvariableop_resource:
ААH
9sequential_lstm_lstm_cell_split_1_readvariableop_resource:	АD
1sequential_lstm_lstm_cell_readvariableop_resource:	@АA
/sequential_dense_matmul_readvariableop_resource:@ >
0sequential_dense_biasadd_readvariableop_resource: C
1sequential_dense_1_matmul_readvariableop_resource: @
2sequential_dense_1_biasadd_readvariableop_resource:
identityИҐ'sequential/dense/BiasAdd/ReadVariableOpҐ&sequential/dense/MatMul/ReadVariableOpҐ)sequential/dense_1/BiasAdd/ReadVariableOpҐ(sequential/dense_1/MatMul/ReadVariableOpҐ(sequential/lstm/lstm_cell/ReadVariableOpҐ*sequential/lstm/lstm_cell/ReadVariableOp_1Ґ*sequential/lstm/lstm_cell/ReadVariableOp_2Ґ*sequential/lstm/lstm_cell/ReadVariableOp_3Ґ.sequential/lstm/lstm_cell/split/ReadVariableOpҐ0sequential/lstm/lstm_cell/split_1/ReadVariableOpҐsequential/lstm/whileҐ9sequential/time_distributed/conv2d/BiasAdd/ReadVariableOpҐ8sequential/time_distributed/conv2d/Conv2D/ReadVariableOpҐ=sequential/time_distributed_1/conv2d_1/BiasAdd/ReadVariableOpҐ<sequential/time_distributed_1/conv2d_1/Conv2D/ReadVariableOpВ
)sequential/time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         і
#sequential/time_distributed/ReshapeReshapetime_distributed_input2sequential/time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€¬
8sequential/time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOpAsequential_time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ж
)sequential/time_distributed/conv2d/Conv2DConv2D,sequential/time_distributed/Reshape:output:0@sequential/time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
Є
9sequential/time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOpBsequential_time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ж
*sequential/time_distributed/conv2d/BiasAddBiasAdd2sequential/time_distributed/conv2d/Conv2D:output:0Asequential/time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Ю
'sequential/time_distributed/conv2d/ReluRelu3sequential/time_distributed/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€И
+sequential/time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€            џ
%sequential/time_distributed/Reshape_1Reshape5sequential/time_distributed/conv2d/Relu:activations:04sequential/time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€Д
+sequential/time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         Є
%sequential/time_distributed/Reshape_2Reshapetime_distributed_input4sequential/time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€Д
+sequential/time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         –
%sequential/time_distributed_1/ReshapeReshape.sequential/time_distributed/Reshape_1:output:04sequential/time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 
<sequential/time_distributed_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpEsequential_time_distributed_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Р
-sequential/time_distributed_1/conv2d_1/Conv2DConv2D.sequential/time_distributed_1/Reshape:output:0Dsequential/time_distributed_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 *
paddingVALID*
strides
ј
=sequential/time_distributed_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpFsequential_time_distributed_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0т
.sequential/time_distributed_1/conv2d_1/BiasAddBiasAdd6sequential/time_distributed_1/conv2d_1/Conv2D:output:0Esequential/time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 ¶
+sequential/time_distributed_1/conv2d_1/ReluRelu7sequential/time_distributed_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 К
-sequential/time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€   	          г
'sequential/time_distributed_1/Reshape_1Reshape9sequential/time_distributed_1/conv2d_1/Relu:activations:06sequential/time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€	 Ж
-sequential/time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ‘
'sequential/time_distributed_1/Reshape_2Reshape.sequential/time_distributed/Reshape_1:output:06sequential/time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€Д
+sequential/time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          “
%sequential/time_distributed_2/ReshapeReshape0sequential/time_distributed_1/Reshape_1:output:04sequential/time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 џ
3sequential/time_distributed_2/max_pooling2d/MaxPoolMaxPool.sequential/time_distributed_2/Reshape:output:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
К
-sequential/time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€             ж
'sequential/time_distributed_2/Reshape_1Reshape<sequential/time_distributed_2/max_pooling2d/MaxPool:output:06sequential/time_distributed_2/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€ Ж
-sequential/time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          ÷
'sequential/time_distributed_2/Reshape_2Reshape0sequential/time_distributed_1/Reshape_1:output:06sequential/time_distributed_2/Reshape_2/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 Д
+sequential/time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          “
%sequential/time_distributed_3/ReshapeReshape0sequential/time_distributed_2/Reshape_1:output:04sequential/time_distributed_3/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ |
+sequential/time_distributed_3/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   —
-sequential/time_distributed_3/flatten/ReshapeReshape.sequential/time_distributed_3/Reshape:output:04sequential/time_distributed_3/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АВ
-sequential/time_distributed_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€      ў
'sequential/time_distributed_3/Reshape_1Reshape6sequential/time_distributed_3/flatten/Reshape:output:06sequential/time_distributed_3/Reshape_1/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€АЖ
-sequential/time_distributed_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          ÷
'sequential/time_distributed_3/Reshape_2Reshape0sequential/time_distributed_2/Reshape_1:output:06sequential/time_distributed_3/Reshape_2/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ u
sequential/lstm/ShapeShape0sequential/time_distributed_3/Reshape_1:output:0*
T0*
_output_shapes
:m
#sequential/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%sequential/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%sequential/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
sequential/lstm/strided_sliceStridedSlicesequential/lstm/Shape:output:0,sequential/lstm/strided_slice/stack:output:0.sequential/lstm/strided_slice/stack_1:output:0.sequential/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
sequential/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@£
sequential/lstm/zeros/packedPack&sequential/lstm/strided_slice:output:0'sequential/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
sequential/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ь
sequential/lstm/zerosFill%sequential/lstm/zeros/packed:output:0$sequential/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@b
 sequential/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@І
sequential/lstm/zeros_1/packedPack&sequential/lstm/strided_slice:output:0)sequential/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
sequential/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ґ
sequential/lstm/zeros_1Fill'sequential/lstm/zeros_1/packed:output:0&sequential/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@s
sequential/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Є
sequential/lstm/transpose	Transpose0sequential/time_distributed_3/Reshape_1:output:0'sequential/lstm/transpose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€Аd
sequential/lstm/Shape_1Shapesequential/lstm/transpose:y:0*
T0*
_output_shapes
:o
%sequential/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'sequential/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'sequential/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ђ
sequential/lstm/strided_slice_1StridedSlice sequential/lstm/Shape_1:output:0.sequential/lstm/strided_slice_1/stack:output:00sequential/lstm/strided_slice_1/stack_1:output:00sequential/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+sequential/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€д
sequential/lstm/TensorArrayV2TensorListReserve4sequential/lstm/TensorArrayV2/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ц
Esequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Р
7sequential/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/lstm/transpose:y:0Nsequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“o
%sequential/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'sequential/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'sequential/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
sequential/lstm/strided_slice_2StridedSlicesequential/lstm/transpose:y:0.sequential/lstm/strided_slice_2/stack:output:00sequential/lstm/strided_slice_2/stack_1:output:00sequential/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_maskw
)sequential/lstm/lstm_cell/ones_like/ShapeShapesequential/lstm/zeros:output:0*
T0*
_output_shapes
:n
)sequential/lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?≈
#sequential/lstm/lstm_cell/ones_likeFill2sequential/lstm/lstm_cell/ones_like/Shape:output:02sequential/lstm/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@k
)sequential/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :®
.sequential/lstm/lstm_cell/split/ReadVariableOpReadVariableOp7sequential_lstm_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0р
sequential/lstm/lstm_cell/splitSplit2sequential/lstm/lstm_cell/split/split_dim:output:06sequential/lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_split∞
 sequential/lstm/lstm_cell/MatMulMatMul(sequential/lstm/strided_slice_2:output:0(sequential/lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@≤
"sequential/lstm/lstm_cell/MatMul_1MatMul(sequential/lstm/strided_slice_2:output:0(sequential/lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@≤
"sequential/lstm/lstm_cell/MatMul_2MatMul(sequential/lstm/strided_slice_2:output:0(sequential/lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@≤
"sequential/lstm/lstm_cell/MatMul_3MatMul(sequential/lstm/strided_slice_2:output:0(sequential/lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@m
+sequential/lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : І
0sequential/lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp9sequential_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0в
!sequential/lstm/lstm_cell/split_1Split4sequential/lstm/lstm_cell/split_1/split_dim:output:08sequential/lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitґ
!sequential/lstm/lstm_cell/BiasAddBiasAdd*sequential/lstm/lstm_cell/MatMul:product:0*sequential/lstm/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ї
#sequential/lstm/lstm_cell/BiasAdd_1BiasAdd,sequential/lstm/lstm_cell/MatMul_1:product:0*sequential/lstm/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@Ї
#sequential/lstm/lstm_cell/BiasAdd_2BiasAdd,sequential/lstm/lstm_cell/MatMul_2:product:0*sequential/lstm/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@Ї
#sequential/lstm/lstm_cell/BiasAdd_3BiasAdd,sequential/lstm/lstm_cell/MatMul_3:product:0*sequential/lstm/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@§
sequential/lstm/lstm_cell/mulMulsequential/lstm/zeros:output:0,sequential/lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@¶
sequential/lstm/lstm_cell/mul_1Mulsequential/lstm/zeros:output:0,sequential/lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@¶
sequential/lstm/lstm_cell/mul_2Mulsequential/lstm/zeros:output:0,sequential/lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@¶
sequential/lstm/lstm_cell/mul_3Mulsequential/lstm/zeros:output:0,sequential/lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ы
(sequential/lstm/lstm_cell/ReadVariableOpReadVariableOp1sequential_lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0~
-sequential/lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        А
/sequential/lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   А
/sequential/lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
'sequential/lstm/lstm_cell/strided_sliceStridedSlice0sequential/lstm/lstm_cell/ReadVariableOp:value:06sequential/lstm/lstm_cell/strided_slice/stack:output:08sequential/lstm/lstm_cell/strided_slice/stack_1:output:08sequential/lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask≥
"sequential/lstm/lstm_cell/MatMul_4MatMul!sequential/lstm/lstm_cell/mul:z:00sequential/lstm/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@≤
sequential/lstm/lstm_cell/addAddV2*sequential/lstm/lstm_cell/BiasAdd:output:0,sequential/lstm/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@Б
!sequential/lstm/lstm_cell/SigmoidSigmoid!sequential/lstm/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Э
*sequential/lstm/lstm_cell/ReadVariableOp_1ReadVariableOp1sequential_lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0А
/sequential/lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   В
1sequential/lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   В
1sequential/lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ч
)sequential/lstm/lstm_cell/strided_slice_1StridedSlice2sequential/lstm/lstm_cell/ReadVariableOp_1:value:08sequential/lstm/lstm_cell/strided_slice_1/stack:output:0:sequential/lstm/lstm_cell/strided_slice_1/stack_1:output:0:sequential/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЈ
"sequential/lstm/lstm_cell/MatMul_5MatMul#sequential/lstm/lstm_cell/mul_1:z:02sequential/lstm/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@ґ
sequential/lstm/lstm_cell/add_1AddV2,sequential/lstm/lstm_cell/BiasAdd_1:output:0,sequential/lstm/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@Е
#sequential/lstm/lstm_cell/Sigmoid_1Sigmoid#sequential/lstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@£
sequential/lstm/lstm_cell/mul_4Mul'sequential/lstm/lstm_cell/Sigmoid_1:y:0 sequential/lstm/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Э
*sequential/lstm/lstm_cell/ReadVariableOp_2ReadVariableOp1sequential_lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0А
/sequential/lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   В
1sequential/lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   В
1sequential/lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ч
)sequential/lstm/lstm_cell/strided_slice_2StridedSlice2sequential/lstm/lstm_cell/ReadVariableOp_2:value:08sequential/lstm/lstm_cell/strided_slice_2/stack:output:0:sequential/lstm/lstm_cell/strided_slice_2/stack_1:output:0:sequential/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЈ
"sequential/lstm/lstm_cell/MatMul_6MatMul#sequential/lstm/lstm_cell/mul_2:z:02sequential/lstm/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@ґ
sequential/lstm/lstm_cell/add_2AddV2,sequential/lstm/lstm_cell/BiasAdd_2:output:0,sequential/lstm/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@}
sequential/lstm/lstm_cell/ReluRelu#sequential/lstm/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@≠
sequential/lstm/lstm_cell/mul_5Mul%sequential/lstm/lstm_cell/Sigmoid:y:0,sequential/lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@§
sequential/lstm/lstm_cell/add_3AddV2#sequential/lstm/lstm_cell/mul_4:z:0#sequential/lstm/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Э
*sequential/lstm/lstm_cell/ReadVariableOp_3ReadVariableOp1sequential_lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0А
/sequential/lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   В
1sequential/lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        В
1sequential/lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ч
)sequential/lstm/lstm_cell/strided_slice_3StridedSlice2sequential/lstm/lstm_cell/ReadVariableOp_3:value:08sequential/lstm/lstm_cell/strided_slice_3/stack:output:0:sequential/lstm/lstm_cell/strided_slice_3/stack_1:output:0:sequential/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЈ
"sequential/lstm/lstm_cell/MatMul_7MatMul#sequential/lstm/lstm_cell/mul_3:z:02sequential/lstm/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@ґ
sequential/lstm/lstm_cell/add_4AddV2,sequential/lstm/lstm_cell/BiasAdd_3:output:0,sequential/lstm/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@Е
#sequential/lstm/lstm_cell/Sigmoid_2Sigmoid#sequential/lstm/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@
 sequential/lstm/lstm_cell/Relu_1Relu#sequential/lstm/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@±
sequential/lstm/lstm_cell/mul_6Mul'sequential/lstm/lstm_cell/Sigmoid_2:y:0.sequential/lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@~
-sequential/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   и
sequential/lstm/TensorArrayV2_1TensorListReserve6sequential/lstm/TensorArrayV2_1/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“V
sequential/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(sequential/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€d
"sequential/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѕ
sequential/lstm/whileWhile+sequential/lstm/while/loop_counter:output:01sequential/lstm/while/maximum_iterations:output:0sequential/lstm/time:output:0(sequential/lstm/TensorArrayV2_1:handle:0sequential/lstm/zeros:output:0 sequential/lstm/zeros_1:output:0(sequential/lstm/strided_slice_1:output:0Gsequential/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_lstm_lstm_cell_split_readvariableop_resource9sequential_lstm_lstm_cell_split_1_readvariableop_resource1sequential_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *-
body%R#
!sequential_lstm_while_body_345023*-
cond%R#
!sequential_lstm_while_cond_345022*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations С
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   т
2sequential/lstm/TensorArrayV2Stack/TensorListStackTensorListStacksequential/lstm/while:output:3Isequential/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0x
%sequential/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€q
'sequential/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'sequential/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:„
sequential/lstm/strided_slice_3StridedSlice;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0.sequential/lstm/strided_slice_3/stack:output:00sequential/lstm/strided_slice_3/stack_1:output:00sequential/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_masku
 sequential/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ∆
sequential/lstm/transpose_1	Transpose;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)sequential/lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@k
sequential/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Ц
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0≠
sequential/dense/MatMulMatMul(sequential/lstm/strided_slice_3:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ф
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0ђ
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
IdentityIdentity#sequential/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ф
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp)^sequential/lstm/lstm_cell/ReadVariableOp+^sequential/lstm/lstm_cell/ReadVariableOp_1+^sequential/lstm/lstm_cell/ReadVariableOp_2+^sequential/lstm/lstm_cell/ReadVariableOp_3/^sequential/lstm/lstm_cell/split/ReadVariableOp1^sequential/lstm/lstm_cell/split_1/ReadVariableOp^sequential/lstm/while:^sequential/time_distributed/conv2d/BiasAdd/ReadVariableOp9^sequential/time_distributed/conv2d/Conv2D/ReadVariableOp>^sequential/time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp=^sequential/time_distributed_1/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:€€€€€€€€€: : : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2T
(sequential/lstm/lstm_cell/ReadVariableOp(sequential/lstm/lstm_cell/ReadVariableOp2X
*sequential/lstm/lstm_cell/ReadVariableOp_1*sequential/lstm/lstm_cell/ReadVariableOp_12X
*sequential/lstm/lstm_cell/ReadVariableOp_2*sequential/lstm/lstm_cell/ReadVariableOp_22X
*sequential/lstm/lstm_cell/ReadVariableOp_3*sequential/lstm/lstm_cell/ReadVariableOp_32`
.sequential/lstm/lstm_cell/split/ReadVariableOp.sequential/lstm/lstm_cell/split/ReadVariableOp2d
0sequential/lstm/lstm_cell/split_1/ReadVariableOp0sequential/lstm/lstm_cell/split_1/ReadVariableOp2.
sequential/lstm/whilesequential/lstm/while2v
9sequential/time_distributed/conv2d/BiasAdd/ReadVariableOp9sequential/time_distributed/conv2d/BiasAdd/ReadVariableOp2t
8sequential/time_distributed/conv2d/Conv2D/ReadVariableOp8sequential/time_distributed/conv2d/Conv2D/ReadVariableOp2~
=sequential/time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp=sequential/time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp2|
<sequential/time_distributed_1/conv2d_1/Conv2D/ReadVariableOp<sequential/time_distributed_1/conv2d_1/Conv2D/ReadVariableOp:k g
3
_output_shapes!
:€€€€€€€€€
0
_user_specified_nametime_distributed_input
Ш

т
A__inference_dense_layer_call_and_return_conditional_losses_346253

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
µ
√
while_cond_348446
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_348446___redundant_placeholder04
0while_while_cond_348446___redundant_placeholder14
0while_while_cond_348446___redundant_placeholder24
0while_while_cond_348446___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
µ
√
while_cond_348185
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_348185___redundant_placeholder04
0while_while_cond_348185___redundant_placeholder14
0while_while_cond_348185___redundant_placeholder24
0while_while_cond_348185___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
оi
В	
while_body_346107
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
АА@
1while_lstm_cell_split_1_readvariableop_resource_0:	А<
)while_lstm_cell_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
АА>
/while_lstm_cell_split_1_readvariableop_resource:	А:
'while_lstm_cell_readvariableop_resource:	@АИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€А*
element_dtype0b
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:d
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?І
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0“
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_split§
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Х
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0ƒ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitШ
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@Е
while/lstm_cell/mulMulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
while/lstm_cell/mul_1Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
while/lstm_cell/mul_2Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
while/lstm_cell/mul_3Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Й
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskХ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ф
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@В
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@i
while/lstm_cell/ReluReluwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@П
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@У
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@¬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@¶

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
Д
э
D__inference_conv2d_1_layer_call_and_return_conditional_losses_345274

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€	 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ж
O
3__inference_time_distributed_2_layer_call_fn_347912

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_346672l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€	 :[ W
3
_output_shapes!
:€€€€€€€€€	 
 
_user_specified_nameinputs
ц
Г
!sequential_lstm_while_cond_345022<
8sequential_lstm_while_sequential_lstm_while_loop_counterB
>sequential_lstm_while_sequential_lstm_while_maximum_iterations%
!sequential_lstm_while_placeholder'
#sequential_lstm_while_placeholder_1'
#sequential_lstm_while_placeholder_2'
#sequential_lstm_while_placeholder_3>
:sequential_lstm_while_less_sequential_lstm_strided_slice_1T
Psequential_lstm_while_sequential_lstm_while_cond_345022___redundant_placeholder0T
Psequential_lstm_while_sequential_lstm_while_cond_345022___redundant_placeholder1T
Psequential_lstm_while_sequential_lstm_while_cond_345022___redundant_placeholder2T
Psequential_lstm_while_sequential_lstm_while_cond_345022___redundant_placeholder3"
sequential_lstm_while_identity
Ґ
sequential/lstm/while/LessLess!sequential_lstm_while_placeholder:sequential_lstm_while_less_sequential_lstm_strided_slice_1*
T0*
_output_shapes
: k
sequential/lstm/while/IdentityIdentitysequential/lstm/while/Less:z:0*
T0
*
_output_shapes
: "I
sequential_lstm_while_identity'sequential/lstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
љ
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_348030

inputs
identityf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   w
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аd
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€      
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€А_
IdentityIdentityReshape_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ :[ W
3
_output_shapes!
:€€€€€€€€€ 
 
_user_specified_nameinputs
Є
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_347930

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 Я
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : Ќ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:Х
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ o
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&€€€€€€€€€€€€€€€€€€	 :d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 
 
_user_specified_nameinputs
і
¶
1__inference_time_distributed_layer_call_fn_347682

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_345242Д
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&€€€€€€€€€€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
љ
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_346653

inputs
identityf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   w
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аd
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€      
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€А_
IdentityIdentityReshape_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ :[ W
3
_output_shapes!
:€€€€€€€€€ 
 
_user_specified_nameinputs
ј
“
L__inference_time_distributed_layer_call_and_return_conditional_losses_345242

inputs'
conv2d_345230:
conv2d_345232:
identityИҐconv2d/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€€
conv2d/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_345230conv2d_345232*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_345188\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Ќ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:Ю
	Reshape_1Reshape'conv2d/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€g
NoOpNoOp^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&€€€€€€€€€€€€€€€€€€: : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ь
O
3__inference_time_distributed_3_layer_call_fn_347976

inputs
identityћ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_345459n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&€€€€€€€€€€€€€€€€€€ :d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Ѕ	
І
lstm_while_cond_347176&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1>
:lstm_while_lstm_while_cond_347176___redundant_placeholder0>
:lstm_while_lstm_while_cond_347176___redundant_placeholder1>
:lstm_while_lstm_while_cond_347176___redundant_placeholder2>
:lstm_while_lstm_while_cond_347176___redundant_placeholder3
lstm_while_identity
v
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: U
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: "3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
З
°
L__inference_time_distributed_layer_call_and_return_conditional_losses_347763

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identityИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€К
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0≤
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Т
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€l
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€            З
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€m
IdentityIdentityReshape_1:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€Е
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
±Р
В	
while_body_348447
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
АА@
1while_lstm_cell_split_1_readvariableop_resource_0:	А<
)while_lstm_cell_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
АА>
/while_lstm_cell_split_1_readvariableop_resource:	А:
'while_lstm_cell_readvariableop_resource:	@АИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€А*
element_dtype0b
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:d
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?І
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@b
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @†
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@o
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
: 
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2§Јnk
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?÷
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@П
while/lstm_cell/dropout/CastCast(while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Щ
while/lstm_cell/dropout/Mul_1Mulwhile/lstm_cell/dropout/Mul:z:0 while/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@d
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @§
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:ѕ
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2≈ХЧm
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?№
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@У
while/lstm_cell/dropout_1/CastCast*while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Я
while/lstm_cell/dropout_1/Mul_1Mul!while/lstm_cell/dropout_1/Mul:z:0"while/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@d
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @§
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:ѕ
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2фаћm
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?№
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@У
while/lstm_cell/dropout_2/CastCast*while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Я
while/lstm_cell/dropout_2/Mul_1Mul!while/lstm_cell/dropout_2/Mul:z:0"while/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@d
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @§
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:ѕ
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2‘…Ёm
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?№
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@У
while/lstm_cell/dropout_3/CastCast*while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Я
while/lstm_cell/dropout_3/Mul_1Mul!while/lstm_cell/dropout_3/Mul:z:0"while/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0“
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_split§
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Х
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0ƒ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitШ
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@Д
while/lstm_cell/mulMulwhile_placeholder_2!while/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@И
while/lstm_cell/mul_1Mulwhile_placeholder_2#while/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@И
while/lstm_cell/mul_2Mulwhile_placeholder_2#while/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@И
while/lstm_cell/mul_3Mulwhile_placeholder_2#while/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Й
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskХ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ф
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@В
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@i
while/lstm_cell/ReluReluwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@П
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@У
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@¬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@¶

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
ч

™
$__inference_signature_wrapper_346980
time_distributed_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: 
	unknown_3:
АА
	unknown_4:	А
	unknown_5:	@А
	unknown_6:@ 
	unknown_7: 
	unknown_8: 
	unknown_9:
identityИҐStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCalltime_distributed_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*-
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В **
f%R#
!__inference__wrapped_model_345163o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:€€€€€€€€€: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:k g
3
_output_shapes!
:€€€€€€€€€
0
_user_specified_nametime_distributed_input
Ѕ
У
&__inference_dense_layer_call_fn_349137

inputs
unknown:@ 
	unknown_0: 
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_346253o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
≤
D
(__inference_flatten_layer_call_fn_349232

inputs
identityі
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_345425a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€ :W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ћ
J
.__inference_max_pooling2d_layer_call_fn_349217

inputs
identityЅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_345365h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€	 :W S
/
_output_shapes
:€€€€€€€€€	 
 
_user_specified_nameinputs
£
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_347957

inputs
identityf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 Я
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
l
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€             М
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€ f
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€	 :[ W
3
_output_shapes!
:€€€€€€€€€	 
 
_user_specified_nameinputs
≈
Х
(__inference_dense_1_layer_call_fn_349157

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_346269o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Д
э
D__inference_conv2d_1_layer_call_and_return_conditional_losses_349207

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€	 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ѕ	
І
lstm_while_cond_347491&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1>
:lstm_while_lstm_while_cond_347491___redundant_placeholder0>
:lstm_while_lstm_while_cond_347491___redundant_placeholder1>
:lstm_while_lstm_while_cond_347491___redundant_placeholder2>
:lstm_while_lstm_while_cond_347491___redundant_placeholder3
lstm_while_identity
v
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: U
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: "3
lstm_while_identitylstm/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
Ў
O
3__inference_time_distributed_3_layer_call_fn_347981

inputs
identity√
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_346002e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ :[ W
3
_output_shapes!
:€€€€€€€€€ 
 
_user_specified_nameinputs
Й]
©
E__inference_lstm_cell_layer_call_and_return_conditional_losses_349454

inputs
states_0
states_11
split_readvariableop_resource:
АА.
split_1_readvariableop_resource:	А*
readvariableop_resource:	@А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґsplit/ReadVariableOpҐsplit_1/ReadVariableOpG
ones_like/ShapeShapestates_0*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:™
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2Ш§][
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¶
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:ѓ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2≈ић]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ђ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@s
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@o
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:ѓ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2йч®]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ђ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@s
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@o
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:ѓ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ИмН]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ђ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@s
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@o
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ґ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitZ
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€@\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€@\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€@\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€@S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Ф
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@Y
mulMulstates_0dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@]
mul_1Mulstates_0dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@]
mul_2Mulstates_0dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@]
mul_3Mulstates_0dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      л
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maske
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€@i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@W
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€@i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@_
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@V
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@K
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@c
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@X
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Z

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@ј
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€А:€€€€€€€€€@:€€€€€€€€€@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/1
і
¶
1__inference_time_distributed_layer_call_fn_347673

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_345201Д
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&€€€€€€€€€€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∆	
ф
C__inference_dense_1_layer_call_and_return_conditional_losses_349167

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
З
°
L__inference_time_distributed_layer_call_and_return_conditional_losses_345951

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identityИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€К
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0≤
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Т
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€l
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€            З
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€m
IdentityIdentityReshape_1:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€Е
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
ТЭ
±

lstm_while_body_347492&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0H
4lstm_while_lstm_cell_split_readvariableop_resource_0:
ААE
6lstm_while_lstm_cell_split_1_readvariableop_resource_0:	АA
.lstm_while_lstm_cell_readvariableop_resource_0:	@А
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorF
2lstm_while_lstm_cell_split_readvariableop_resource:
ААC
4lstm_while_lstm_cell_split_1_readvariableop_resource:	А?
,lstm_while_lstm_cell_readvariableop_resource:	@АИҐ#lstm/while/lstm_cell/ReadVariableOpҐ%lstm/while/lstm_cell/ReadVariableOp_1Ґ%lstm/while/lstm_cell/ReadVariableOp_2Ґ%lstm/while/lstm_cell/ReadVariableOp_3Ґ)lstm/while/lstm_cell/split/ReadVariableOpҐ+lstm/while/lstm_cell/split_1/ReadVariableOpН
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ј
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€А*
element_dtype0l
$lstm/while/lstm_cell/ones_like/ShapeShapelstm_while_placeholder_2*
T0*
_output_shapes
:i
$lstm/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ґ
lstm/while/lstm_cell/ones_likeFill-lstm/while/lstm_cell/ones_like/Shape:output:0-lstm/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@g
"lstm/while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @ѓ
 lstm/while/lstm_cell/dropout/MulMul'lstm/while/lstm_cell/ones_like:output:0+lstm/while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@y
"lstm/while/lstm_cell/dropout/ShapeShape'lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:’
9lstm/while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform+lstm/while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ат£p
+lstm/while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?е
)lstm/while/lstm_cell/dropout/GreaterEqualGreaterEqualBlstm/while/lstm_cell/dropout/random_uniform/RandomUniform:output:04lstm/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Щ
!lstm/while/lstm_cell/dropout/CastCast-lstm/while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@®
"lstm/while/lstm_cell/dropout/Mul_1Mul$lstm/while/lstm_cell/dropout/Mul:z:0%lstm/while/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@i
$lstm/while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @≥
"lstm/while/lstm_cell/dropout_1/MulMul'lstm/while/lstm_cell/ones_like:output:0-lstm/while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@{
$lstm/while/lstm_cell/dropout_1/ShapeShape'lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:ў
;lstm/while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2Л±–r
-lstm/while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?л
+lstm/while/lstm_cell/dropout_1/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Э
#lstm/while/lstm_cell/dropout_1/CastCast/lstm/while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Ѓ
$lstm/while/lstm_cell/dropout_1/Mul_1Mul&lstm/while/lstm_cell/dropout_1/Mul:z:0'lstm/while/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@i
$lstm/while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @≥
"lstm/while/lstm_cell/dropout_2/MulMul'lstm/while/lstm_cell/ones_like:output:0-lstm/while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@{
$lstm/while/lstm_cell/dropout_2/ShapeShape'lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:ў
;lstm/while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2»шгr
-lstm/while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?л
+lstm/while/lstm_cell/dropout_2/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Э
#lstm/while/lstm_cell/dropout_2/CastCast/lstm/while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Ѓ
$lstm/while/lstm_cell/dropout_2/Mul_1Mul&lstm/while/lstm_cell/dropout_2/Mul:z:0'lstm/while/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@i
$lstm/while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @≥
"lstm/while/lstm_cell/dropout_3/MulMul'lstm/while/lstm_cell/ones_like:output:0-lstm/while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@{
$lstm/while/lstm_cell/dropout_3/ShapeShape'lstm/while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:ў
;lstm/while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform-lstm/while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2яЕЮr
-lstm/while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?л
+lstm/while/lstm_cell/dropout_3/GreaterEqualGreaterEqualDlstm/while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:06lstm/while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Э
#lstm/while/lstm_cell/dropout_3/CastCast/lstm/while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Ѓ
$lstm/while/lstm_cell/dropout_3/Mul_1Mul&lstm/while/lstm_cell/dropout_3/Mul:z:0'lstm/while/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@f
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :†
)lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp4lstm_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0б
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:01lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_split≥
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@µ
lstm/while/lstm_cell/MatMul_1MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@µ
lstm/while/lstm_cell/MatMul_2MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@µ
lstm/while/lstm_cell/MatMul_3MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@h
&lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Я
+lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0”
lstm/while/lstm_cell/split_1Split/lstm/while/lstm_cell/split_1/split_dim:output:03lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitІ
lstm/while/lstm_cell/BiasAddBiasAdd%lstm/while/lstm_cell/MatMul:product:0%lstm/while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ђ
lstm/while/lstm_cell/BiasAdd_1BiasAdd'lstm/while/lstm_cell/MatMul_1:product:0%lstm/while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@Ђ
lstm/while/lstm_cell/BiasAdd_2BiasAdd'lstm/while/lstm_cell/MatMul_2:product:0%lstm/while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@Ђ
lstm/while/lstm_cell/BiasAdd_3BiasAdd'lstm/while/lstm_cell/MatMul_3:product:0%lstm/while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@У
lstm/while/lstm_cell/mulMullstm_while_placeholder_2&lstm/while/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ч
lstm/while/lstm_cell/mul_1Mullstm_while_placeholder_2(lstm/while/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ч
lstm/while/lstm_cell/mul_2Mullstm_while_placeholder_2(lstm/while/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ч
lstm/while/lstm_cell/mul_3Mullstm_while_placeholder_2(lstm/while/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@У
#lstm/while/lstm_cell/ReadVariableOpReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   {
*lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ‘
"lstm/while/lstm_cell/strided_sliceStridedSlice+lstm/while/lstm_cell/ReadVariableOp:value:01lstm/while/lstm_cell/strided_slice/stack:output:03lstm/while/lstm_cell/strided_slice/stack_1:output:03lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask§
lstm/while/lstm_cell/MatMul_4MatMullstm/while/lstm_cell/mul:z:0+lstm/while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@£
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/BiasAdd:output:0'lstm/while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@w
lstm/while/lstm_cell/SigmoidSigmoidlstm/while/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Х
%lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0{
*lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   }
,lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   }
,lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ё
$lstm/while/lstm_cell/strided_slice_1StridedSlice-lstm/while/lstm_cell/ReadVariableOp_1:value:03lstm/while/lstm_cell/strided_slice_1/stack:output:05lstm/while/lstm_cell/strided_slice_1/stack_1:output:05lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask®
lstm/while/lstm_cell/MatMul_5MatMullstm/while/lstm_cell/mul_1:z:0-lstm/while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@І
lstm/while/lstm_cell/add_1AddV2'lstm/while/lstm_cell/BiasAdd_1:output:0'lstm/while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@{
lstm/while/lstm_cell/Sigmoid_1Sigmoidlstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@С
lstm/while/lstm_cell/mul_4Mul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@Х
%lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0{
*lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   }
,lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   }
,lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ё
$lstm/while/lstm_cell/strided_slice_2StridedSlice-lstm/while/lstm_cell/ReadVariableOp_2:value:03lstm/while/lstm_cell/strided_slice_2/stack:output:05lstm/while/lstm_cell/strided_slice_2/stack_1:output:05lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask®
lstm/while/lstm_cell/MatMul_6MatMullstm/while/lstm_cell/mul_2:z:0-lstm/while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@І
lstm/while/lstm_cell/add_2AddV2'lstm/while/lstm_cell/BiasAdd_2:output:0'lstm/while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@s
lstm/while/lstm_cell/ReluRelulstm/while/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ю
lstm/while/lstm_cell/mul_5Mul lstm/while/lstm_cell/Sigmoid:y:0'lstm/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@Х
lstm/while/lstm_cell/add_3AddV2lstm/while/lstm_cell/mul_4:z:0lstm/while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Х
%lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0{
*lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   }
,lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ё
$lstm/while/lstm_cell/strided_slice_3StridedSlice-lstm/while/lstm_cell/ReadVariableOp_3:value:03lstm/while/lstm_cell/strided_slice_3/stack:output:05lstm/while/lstm_cell/strided_slice_3/stack_1:output:05lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask®
lstm/while/lstm_cell/MatMul_7MatMullstm/while/lstm_cell/mul_3:z:0-lstm/while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@І
lstm/while/lstm_cell/add_4AddV2'lstm/while/lstm_cell/BiasAdd_3:output:0'lstm/while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@{
lstm/while/lstm_cell/Sigmoid_2Sigmoidlstm/while/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@u
lstm/while/lstm_cell/Relu_1Relulstm/while/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ґ
lstm/while/lstm_cell/mul_6Mul"lstm/while/lstm_cell/Sigmoid_2:y:0)lstm/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@÷
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype0:йи“R
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: T
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: h
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ~
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: h
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: Х
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: Е
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_6:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Е
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_3:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@…
lstm/while/NoOpNoOp$^lstm/while/lstm_cell/ReadVariableOp&^lstm/while/lstm_cell/ReadVariableOp_1&^lstm/while/lstm_cell/ReadVariableOp_2&^lstm/while/lstm_cell/ReadVariableOp_3*^lstm/while/lstm_cell/split/ReadVariableOp,^lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"^
,lstm_while_lstm_cell_readvariableop_resource.lstm_while_lstm_cell_readvariableop_resource_0"n
4lstm_while_lstm_cell_split_1_readvariableop_resource6lstm_while_lstm_cell_split_1_readvariableop_resource_0"j
2lstm_while_lstm_cell_split_readvariableop_resource4lstm_while_lstm_cell_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"Љ
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2J
#lstm/while/lstm_cell/ReadVariableOp#lstm/while/lstm_cell/ReadVariableOp2N
%lstm/while/lstm_cell/ReadVariableOp_1%lstm/while/lstm_cell/ReadVariableOp_12N
%lstm/while/lstm_cell/ReadVariableOp_2%lstm/while/lstm_cell/ReadVariableOp_22N
%lstm/while/lstm_cell/ReadVariableOp_3%lstm/while/lstm_cell/ReadVariableOp_32V
)lstm/while/lstm_cell/split/ReadVariableOp)lstm/while/lstm_cell/split/ReadVariableOp2Z
+lstm/while/lstm_cell/split_1/ReadVariableOp+lstm/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
¶"
–
while_body_345586
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_345610_0:
АА'
while_lstm_cell_345612_0:	А+
while_lstm_cell_345614_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_345610:
АА%
while_lstm_cell_345612:	А)
while_lstm_cell_345614:	@АИҐ'while/lstm_cell/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€А*
element_dtype0©
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_345610_0while_lstm_cell_345612_0while_lstm_cell_345614_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_345572ў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Н
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Н
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@v

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"2
while_lstm_cell_345610while_lstm_cell_345610_0"2
while_lstm_cell_345612while_lstm_cell_345612_0"2
while_lstm_cell_345614while_lstm_cell_345614_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
є
J
.__inference_max_pooling2d_layer_call_fn_349212

inputs
identity№
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_345344Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
н
ф
*__inference_lstm_cell_layer_call_fn_349255

inputs
states_0
states_1
unknown:
АА
	unknown_0:	А
	unknown_1:	@А
identity

identity_1

identity_2ИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_345572o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€А:€€€€€€€€€@:€€€€€€€€€@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/1
Ў
O
3__inference_time_distributed_3_layer_call_fn_347986

inputs
identity√
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_346653e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ :[ W
3
_output_shapes!
:€€€€€€€€€ 
 
_user_specified_nameinputs
№»
в

F__inference_sequential_layer_call_and_return_conditional_losses_347317

inputsP
6time_distributed_conv2d_conv2d_readvariableop_resource:E
7time_distributed_conv2d_biasadd_readvariableop_resource:T
:time_distributed_1_conv2d_1_conv2d_readvariableop_resource: I
;time_distributed_1_conv2d_1_biasadd_readvariableop_resource: @
,lstm_lstm_cell_split_readvariableop_resource:
АА=
.lstm_lstm_cell_split_1_readvariableop_resource:	А9
&lstm_lstm_cell_readvariableop_resource:	@А6
$dense_matmul_readvariableop_resource:@ 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐlstm/lstm_cell/ReadVariableOpҐlstm/lstm_cell/ReadVariableOp_1Ґlstm/lstm_cell/ReadVariableOp_2Ґlstm/lstm_cell/ReadVariableOp_3Ґ#lstm/lstm_cell/split/ReadVariableOpҐ%lstm/lstm_cell/split_1/ReadVariableOpҐ
lstm/whileҐ.time_distributed/conv2d/BiasAdd/ReadVariableOpҐ-time_distributed/conv2d/Conv2D/ReadVariableOpҐ2time_distributed_1/conv2d_1/BiasAdd/ReadVariableOpҐ1time_distributed_1/conv2d_1/Conv2D/ReadVariableOpw
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         О
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ђ
-time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOp6time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0е
time_distributed/conv2d/Conv2DConv2D!time_distributed/Reshape:output:05time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
Ґ
.time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOp7time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0≈
time_distributed/conv2d/BiasAddBiasAdd'time_distributed/conv2d/Conv2D:output:06time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€И
time_distributed/conv2d/ReluRelu(time_distributed/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€}
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€            Ї
time_distributed/Reshape_1Reshape*time_distributed/conv2d/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€y
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         Т
time_distributed/Reshape_2Reshapeinputs)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€y
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ѓ
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€і
1time_distributed_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp:time_distributed_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0п
"time_distributed_1/conv2d_1/Conv2DConv2D#time_distributed_1/Reshape:output:09time_distributed_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 *
paddingVALID*
strides
™
2time_distributed_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp;time_distributed_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0—
#time_distributed_1/conv2d_1/BiasAddBiasAdd+time_distributed_1/conv2d_1/Conv2D:output:0:time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 Р
 time_distributed_1/conv2d_1/ReluRelu,time_distributed_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€   	          ¬
time_distributed_1/Reshape_1Reshape.time_distributed_1/conv2d_1/Relu:activations:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€	 {
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ≥
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          ±
time_distributed_2/ReshapeReshape%time_distributed_1/Reshape_1:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 ≈
(time_distributed_2/max_pooling2d/MaxPoolMaxPool#time_distributed_2/Reshape:output:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides

"time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€             ≈
time_distributed_2/Reshape_1Reshape1time_distributed_2/max_pooling2d/MaxPool:output:0+time_distributed_2/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€ {
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          µ
time_distributed_2/Reshape_2Reshape%time_distributed_1/Reshape_1:output:0+time_distributed_2/Reshape_2/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 y
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          ±
time_distributed_3/ReshapeReshape%time_distributed_2/Reshape_1:output:0)time_distributed_3/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ q
 time_distributed_3/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ∞
"time_distributed_3/flatten/ReshapeReshape#time_distributed_3/Reshape:output:0)time_distributed_3/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аw
"time_distributed_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€      Є
time_distributed_3/Reshape_1Reshape+time_distributed_3/flatten/Reshape:output:0+time_distributed_3/Reshape_1/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€А{
"time_distributed_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          µ
time_distributed_3/Reshape_2Reshape%time_distributed_2/Reshape_1:output:0+time_distributed_3/Reshape_2/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ _

lstm/ShapeShape%time_distributed_3/Reshape_1:output:0*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@В
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@W
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@Ж
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Б
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@h
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
lstm/transpose	Transpose%time_distributed_3/Reshape_1:output:0lstm/transpose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€АN
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€√
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Л
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   п
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“d
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_maska
lstm/lstm_cell/ones_like/ShapeShapelstm/zeros:output:0*
T0*
_output_shapes
:c
lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?§
lstm/lstm_cell/ones_likeFill'lstm/lstm_cell/ones_like/Shape:output:0'lstm/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@`
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
#lstm/lstm_cell/split/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0ѕ
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0+lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitП
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@С
lstm/lstm_cell/MatMul_1MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@С
lstm/lstm_cell/MatMul_2MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@С
lstm/lstm_cell/MatMul_3MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@b
 lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : С
%lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Ѕ
lstm/lstm_cell/split_1Split)lstm/lstm_cell/split_1/split_dim:output:0-lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitХ
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/MatMul:product:0lstm/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Щ
lstm/lstm_cell/BiasAdd_1BiasAdd!lstm/lstm_cell/MatMul_1:product:0lstm/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@Щ
lstm/lstm_cell/BiasAdd_2BiasAdd!lstm/lstm_cell/MatMul_2:product:0lstm/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@Щ
lstm/lstm_cell/BiasAdd_3BiasAdd!lstm/lstm_cell/MatMul_3:product:0lstm/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@Г
lstm/lstm_cell/mulMullstm/zeros:output:0!lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Е
lstm/lstm_cell/mul_1Mullstm/zeros:output:0!lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Е
lstm/lstm_cell/mul_2Mullstm/zeros:output:0!lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Е
lstm/lstm_cell/mul_3Mullstm/zeros:output:0!lstm/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Е
lstm/lstm_cell/ReadVariableOpReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   u
$lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ґ
lstm/lstm_cell/strided_sliceStridedSlice%lstm/lstm_cell/ReadVariableOp:value:0+lstm/lstm_cell/strided_slice/stack:output:0-lstm/lstm_cell/strided_slice/stack_1:output:0-lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskТ
lstm/lstm_cell/MatMul_4MatMullstm/lstm_cell/mul:z:0%lstm/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@С
lstm/lstm_cell/addAddV2lstm/lstm_cell/BiasAdd:output:0!lstm/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@k
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@З
lstm/lstm_cell/ReadVariableOp_1ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0u
$lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   w
&lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   w
&lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
lstm/lstm_cell/strided_slice_1StridedSlice'lstm/lstm_cell/ReadVariableOp_1:value:0-lstm/lstm_cell/strided_slice_1/stack:output:0/lstm/lstm_cell/strided_slice_1/stack_1:output:0/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЦ
lstm/lstm_cell/MatMul_5MatMullstm/lstm_cell/mul_1:z:0'lstm/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Х
lstm/lstm_cell/add_1AddV2!lstm/lstm_cell/BiasAdd_1:output:0!lstm/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@o
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm/lstm_cell/mul_4Mullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
lstm/lstm_cell/ReadVariableOp_2ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0u
$lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   w
&lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   w
&lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
lstm/lstm_cell/strided_slice_2StridedSlice'lstm/lstm_cell/ReadVariableOp_2:value:0-lstm/lstm_cell/strided_slice_2/stack:output:0/lstm/lstm_cell/strided_slice_2/stack_1:output:0/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЦ
lstm/lstm_cell/MatMul_6MatMullstm/lstm_cell/mul_2:z:0'lstm/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Х
lstm/lstm_cell/add_2AddV2!lstm/lstm_cell/BiasAdd_2:output:0!lstm/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@g
lstm/lstm_cell/ReluRelulstm/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@М
lstm/lstm_cell/mul_5Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@Г
lstm/lstm_cell/add_3AddV2lstm/lstm_cell/mul_4:z:0lstm/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@З
lstm/lstm_cell/ReadVariableOp_3ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0u
$lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   w
&lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
lstm/lstm_cell/strided_slice_3StridedSlice'lstm/lstm_cell/ReadVariableOp_3:value:0-lstm/lstm_cell/strided_slice_3/stack:output:0/lstm/lstm_cell/strided_slice_3/stack_1:output:0/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЦ
lstm/lstm_cell/MatMul_7MatMullstm/lstm_cell/mul_3:z:0'lstm/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Х
lstm/lstm_cell/add_4AddV2!lstm/lstm_cell/BiasAdd_3:output:0!lstm/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@o
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@i
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Р
lstm/lstm_cell/mul_6Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@s
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   «
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“K
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Y
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : µ

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_lstm_cell_split_readvariableop_resource.lstm_lstm_cell_split_1_readvariableop_resource&lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *"
bodyR
lstm_while_body_347177*"
condR
lstm_while_cond_347176*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations Ж
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   —
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0m
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€f
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:†
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          •
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@`
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0М
dense/MatMulMatMullstm/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Л
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€п
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^lstm/lstm_cell/ReadVariableOp ^lstm/lstm_cell/ReadVariableOp_1 ^lstm/lstm_cell/ReadVariableOp_2 ^lstm/lstm_cell/ReadVariableOp_3$^lstm/lstm_cell/split/ReadVariableOp&^lstm/lstm_cell/split_1/ReadVariableOp^lstm/while/^time_distributed/conv2d/BiasAdd/ReadVariableOp.^time_distributed/conv2d/Conv2D/ReadVariableOp3^time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp2^time_distributed_1/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:€€€€€€€€€: : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2>
lstm/lstm_cell/ReadVariableOplstm/lstm_cell/ReadVariableOp2B
lstm/lstm_cell/ReadVariableOp_1lstm/lstm_cell/ReadVariableOp_12B
lstm/lstm_cell/ReadVariableOp_2lstm/lstm_cell/ReadVariableOp_22B
lstm/lstm_cell/ReadVariableOp_3lstm/lstm_cell/ReadVariableOp_32J
#lstm/lstm_cell/split/ReadVariableOp#lstm/lstm_cell/split/ReadVariableOp2N
%lstm/lstm_cell/split_1/ReadVariableOp%lstm/lstm_cell/split_1/ReadVariableOp2

lstm/while
lstm/while2`
.time_distributed/conv2d/BiasAdd/ReadVariableOp.time_distributed/conv2d/BiasAdd/ReadVariableOp2^
-time_distributed/conv2d/Conv2D/ReadVariableOp-time_distributed/conv2d/Conv2D/ReadVariableOp2h
2time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp2time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp2f
1time_distributed_1/conv2d_1/Conv2D/ReadVariableOp1time_distributed_1/conv2d_1/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
±Р
В	
while_body_346468
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
АА@
1while_lstm_cell_split_1_readvariableop_resource_0:	А<
)while_lstm_cell_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
АА>
/while_lstm_cell_split_1_readvariableop_resource:	А:
'while_lstm_cell_readvariableop_resource:	@АИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€А*
element_dtype0b
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:d
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?І
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@b
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @†
while/lstm_cell/dropout/MulMul"while/lstm_cell/ones_like:output:0&while/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@o
while/lstm_cell/dropout/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:Ћ
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2“∆Ъk
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?÷
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@П
while/lstm_cell/dropout/CastCast(while/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Щ
while/lstm_cell/dropout/Mul_1Mulwhile/lstm_cell/dropout/Mul:z:0 while/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@d
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @§
while/lstm_cell/dropout_1/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/dropout_1/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:ѕ
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2яэвm
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?№
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@У
while/lstm_cell/dropout_1/CastCast*while/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Я
while/lstm_cell/dropout_1/Mul_1Mul!while/lstm_cell/dropout_1/Mul:z:0"while/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@d
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @§
while/lstm_cell/dropout_2/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/dropout_2/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:ѕ
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2Вішm
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?№
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@У
while/lstm_cell/dropout_2/CastCast*while/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Я
while/lstm_cell/dropout_2/Mul_1Mul!while/lstm_cell/dropout_2/Mul:z:0"while/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@d
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @§
while/lstm_cell/dropout_3/MulMul"while/lstm_cell/ones_like:output:0(while/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/dropout_3/ShapeShape"while/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:ќ
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2Ртmm
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?№
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@У
while/lstm_cell/dropout_3/CastCast*while/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Я
while/lstm_cell/dropout_3/Mul_1Mul!while/lstm_cell/dropout_3/Mul:z:0"while/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0“
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_split§
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Х
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0ƒ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitШ
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@Д
while/lstm_cell/mulMulwhile_placeholder_2!while/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@И
while/lstm_cell/mul_1Mulwhile_placeholder_2#while/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@И
while/lstm_cell/mul_2Mulwhile_placeholder_2#while/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@И
while/lstm_cell/mul_3Mulwhile_placeholder_2#while/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Й
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskХ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ф
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@В
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@i
while/lstm_cell/ReluReluwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@П
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@У
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@¬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@¶

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
§w
ћ
@__inference_lstm_layer_call_and_return_conditional_losses_348313
inputs_0;
'lstm_cell_split_readvariableop_resource:
АА8
)lstm_cell_split_1_readvariableop_resource:	А4
!lstm_cell_readvariableop_resource:	@А
identityИҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€АD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_maskW
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Х
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0ј
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitА
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : З
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0≤
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitЖ
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@t
lstm_cell/mulMulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@v
lstm_cell/mul_1Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@v
lstm_cell/mul_2Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@v
lstm_cell/mul_3Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@{
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Э
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskГ
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@s
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@]
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@t
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@_
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Б
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : п
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_348186*
condR
while_cond_348185*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Д
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€А: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А
"
_user_specified_name
inputs/0
П
¶
1__inference_time_distributed_layer_call_fn_347691

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_345951{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
К
O
3__inference_time_distributed_2_layer_call_fn_347902

inputs
identity”
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_345402u
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&€€€€€€€€€€€€€€€€€€	 :d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 
 
_user_specified_nameinputs
Є
®
3__inference_time_distributed_1_layer_call_fn_347787

inputs!
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_345287Д
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&€€€€€€€€€€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≈
_
C__inference_flatten_layer_call_and_return_conditional_losses_345425

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€ :W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
пv
 
@__inference_lstm_layer_call_and_return_conditional_losses_346234

inputs;
'lstm_cell_split_readvariableop_resource:
АА8
)lstm_cell_split_1_readvariableop_resource:	А4
!lstm_cell_readvariableop_resource:	@А
identityИҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€АD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_maskW
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Х
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0ј
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitА
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : З
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0≤
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitЖ
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@t
lstm_cell/mulMulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@v
lstm_cell/mul_1Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@v
lstm_cell/mul_2Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@v
lstm_cell/mul_3Mulzeros:output:0lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@{
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Э
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskГ
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@s
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@]
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@t
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@_
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Б
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : п
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_346107*
condR
while_cond_346106*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   ¬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Д
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€А: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
З-
ѓ
F__inference_sequential_layer_call_and_return_conditional_losses_346276

inputs1
time_distributed_345952:%
time_distributed_345954:3
time_distributed_1_345974: '
time_distributed_1_345976: 
lstm_346235:
АА
lstm_346237:	А
lstm_346239:	@А
dense_346254:@ 
dense_346256:  
dense_1_346270: 
dense_1_346272:
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐlstm/StatefulPartitionedCallҐ(time_distributed/StatefulPartitionedCallҐ*time_distributed_1/StatefulPartitionedCall°
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_345952time_distributed_345954*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_345951w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         О
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€‘
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_345974time_distributed_1_345976*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€	 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_345973y
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         љ
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€К
"time_distributed_2/PartitionedCallPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_345989y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          њ
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 ы
"time_distributed_3/PartitionedCallPartitionedCall+time_distributed_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_346002y
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          Ј
time_distributed_3/ReshapeReshape+time_distributed_2/PartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ Щ
lstm/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0lstm_346235lstm_346237lstm_346239*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_346234И
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_346254dense_346256*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_346253С
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_346270dense_1_346272*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_346269w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€€
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:€€€€€€€€€: : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
љ
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_346002

inputs
identityf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   w
flatten/ReshapeReshapeReshape:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аd
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€      
	Reshape_1Reshapeflatten/Reshape:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€А_
IdentityIdentityReshape_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ :[ W
3
_output_shapes!
:€€€€€€€€€ 
 
_user_specified_nameinputs
и7
ч
@__inference_lstm_layer_call_and_return_conditional_losses_345921

inputs$
lstm_cell_345839:
АА
lstm_cell_345841:	А#
lstm_cell_345843:	@А
identityИҐ!lstm_cell/StatefulPartitionedCallҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€АD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_maskл
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_345839lstm_cell_345841lstm_cell_345843*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_345793n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_345839lstm_cell_345841lstm_cell_345843*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_345852*
condR
while_cond_345851*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@r
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€А: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
С
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_345344

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
п
Ю
)__inference_conv2d_1_layer_call_fn_349196

inputs!
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€	 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_345274w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€	 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ь
O
3__inference_time_distributed_3_layer_call_fn_347971

inputs
identityћ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_345432n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&€€€€€€€€€€€€€€€€€€ :d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
ъ
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_345402

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 ў
max_pooling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_345365\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : Ќ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:Э
	Reshape_1Reshape&max_pooling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ o
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&€€€€€€€€€€€€€€€€€€	 :d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 
 
_user_specified_nameinputs
µ
√
while_cond_345851
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_345851___redundant_placeholder04
0while_while_cond_345851___redundant_placeholder14
0while_while_cond_345851___redundant_placeholder24
0while_while_cond_345851___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
оi
В	
while_body_348708
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
АА@
1while_lstm_cell_split_1_readvariableop_resource_0:	А<
)while_lstm_cell_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
АА>
/while_lstm_cell_split_1_readvariableop_resource:	А:
'while_lstm_cell_readvariableop_resource:	@АИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€А*
element_dtype0b
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:d
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?І
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0“
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_split§
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Х
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0ƒ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitШ
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@Е
while/lstm_cell/mulMulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
while/lstm_cell/mul_1Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
while/lstm_cell/mul_2Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
while/lstm_cell/mul_3Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Й
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskХ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ф
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@В
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@i
while/lstm_cell/ReluReluwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@П
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@У
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@¬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@¶

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
 
Ђ
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_347862

inputsA
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: 
identityИҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€О
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ґ
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 *
paddingVALID*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 \
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : Ќ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:Т
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 Й
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&€€€€€€€€€€€€€€€€€€: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
–Ъ
ћ
@__inference_lstm_layer_call_and_return_conditional_losses_348606
inputs_0;
'lstm_cell_split_readvariableop_resource:
АА8
)lstm_cell_split_1_readvariableop_resource:	А4
!lstm_cell_readvariableop_resource:	@А
identityИҐlstm_cell/ReadVariableOpҐlstm_cell/ReadVariableOp_1Ґlstm_cell/ReadVariableOp_2Ґlstm_cell/ReadVariableOp_3Ґlstm_cell/split/ReadVariableOpҐ lstm_cell/split_1/ReadVariableOpҐwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€АD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_maskW
lstm_cell/ones_like/ShapeShapezeros:output:0*
T0*
_output_shapes
:^
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Х
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@\
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @О
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@c
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:њ
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ДРсe
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ƒ
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Г
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@З
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@^
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Т
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:√
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2Й™ѓg
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Н
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@^
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Т
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:√
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ы≤Лg
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Н
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@^
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Т
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:√
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2БЯЧg
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Н
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0ј
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitА
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : З
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0≤
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitЖ
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@К
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@s
lstm_cell/mulMulzeros:output:0lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@w
lstm_cell/mul_1Mulzeros:output:0lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@w
lstm_cell/mul_2Mulzeros:output:0lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@w
lstm_cell/mul_3Mulzeros:output:0lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@{
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Э
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskГ
lstm_cell/MatMul_4MatMullstm_cell/mul:z:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@a
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_5MatMullstm_cell/mul_1:z:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@s
lstm_cell/mul_4Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_6MatMullstm_cell/mul_2:z:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@]
lstm_cell/ReluRelulstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/mul_5Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@t
lstm_cell/add_3AddV2lstm_cell/mul_4:z:0lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      І
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЗ
lstm_cell/MatMul_7MatMullstm_cell/mul_3:z:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@e
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@_
lstm_cell/Relu_1Relulstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Б
lstm_cell/mul_6Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : п
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_348447*
condR
while_cond_348446*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Д
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€А: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А
"
_user_specified_name
inputs/0
т
j
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_345432

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ∆
flatten/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_345425\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :АХ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:Р
	Reshape_1Reshape flatten/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€Аh
IdentityIdentityReshape_1:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&€€€€€€€€€€€€€€€€€€ :d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
©К
≤
!sequential_lstm_while_body_345023<
8sequential_lstm_while_sequential_lstm_while_loop_counterB
>sequential_lstm_while_sequential_lstm_while_maximum_iterations%
!sequential_lstm_while_placeholder'
#sequential_lstm_while_placeholder_1'
#sequential_lstm_while_placeholder_2'
#sequential_lstm_while_placeholder_3;
7sequential_lstm_while_sequential_lstm_strided_slice_1_0w
ssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0S
?sequential_lstm_while_lstm_cell_split_readvariableop_resource_0:
ААP
Asequential_lstm_while_lstm_cell_split_1_readvariableop_resource_0:	АL
9sequential_lstm_while_lstm_cell_readvariableop_resource_0:	@А"
sequential_lstm_while_identity$
 sequential_lstm_while_identity_1$
 sequential_lstm_while_identity_2$
 sequential_lstm_while_identity_3$
 sequential_lstm_while_identity_4$
 sequential_lstm_while_identity_59
5sequential_lstm_while_sequential_lstm_strided_slice_1u
qsequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensorQ
=sequential_lstm_while_lstm_cell_split_readvariableop_resource:
ААN
?sequential_lstm_while_lstm_cell_split_1_readvariableop_resource:	АJ
7sequential_lstm_while_lstm_cell_readvariableop_resource:	@АИҐ.sequential/lstm/while/lstm_cell/ReadVariableOpҐ0sequential/lstm/while/lstm_cell/ReadVariableOp_1Ґ0sequential/lstm/while/lstm_cell/ReadVariableOp_2Ґ0sequential/lstm/while/lstm_cell/ReadVariableOp_3Ґ4sequential/lstm/while/lstm_cell/split/ReadVariableOpҐ6sequential/lstm/while/lstm_cell/split_1/ReadVariableOpШ
Gsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ч
9sequential/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0!sequential_lstm_while_placeholderPsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€А*
element_dtype0В
/sequential/lstm/while/lstm_cell/ones_like/ShapeShape#sequential_lstm_while_placeholder_2*
T0*
_output_shapes
:t
/sequential/lstm/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?„
)sequential/lstm/while/lstm_cell/ones_likeFill8sequential/lstm/while/lstm_cell/ones_like/Shape:output:08sequential/lstm/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@q
/sequential/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ґ
4sequential/lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp?sequential_lstm_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0В
%sequential/lstm/while/lstm_cell/splitSplit8sequential/lstm/while/lstm_cell/split/split_dim:output:0<sequential/lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_split‘
&sequential/lstm/while/lstm_cell/MatMulMatMul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.sequential/lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@÷
(sequential/lstm/while/lstm_cell/MatMul_1MatMul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.sequential/lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@÷
(sequential/lstm/while/lstm_cell/MatMul_2MatMul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.sequential/lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@÷
(sequential/lstm/while/lstm_cell/MatMul_3MatMul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.sequential/lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@s
1sequential/lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : µ
6sequential/lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOpAsequential_lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0ф
'sequential/lstm/while/lstm_cell/split_1Split:sequential/lstm/while/lstm_cell/split_1/split_dim:output:0>sequential/lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split»
'sequential/lstm/while/lstm_cell/BiasAddBiasAdd0sequential/lstm/while/lstm_cell/MatMul:product:00sequential/lstm/while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@ћ
)sequential/lstm/while/lstm_cell/BiasAdd_1BiasAdd2sequential/lstm/while/lstm_cell/MatMul_1:product:00sequential/lstm/while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@ћ
)sequential/lstm/while/lstm_cell/BiasAdd_2BiasAdd2sequential/lstm/while/lstm_cell/MatMul_2:product:00sequential/lstm/while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@ћ
)sequential/lstm/while/lstm_cell/BiasAdd_3BiasAdd2sequential/lstm/while/lstm_cell/MatMul_3:product:00sequential/lstm/while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@µ
#sequential/lstm/while/lstm_cell/mulMul#sequential_lstm_while_placeholder_22sequential/lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ј
%sequential/lstm/while/lstm_cell/mul_1Mul#sequential_lstm_while_placeholder_22sequential/lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ј
%sequential/lstm/while/lstm_cell/mul_2Mul#sequential_lstm_while_placeholder_22sequential/lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ј
%sequential/lstm/while/lstm_cell/mul_3Mul#sequential_lstm_while_placeholder_22sequential/lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@©
.sequential/lstm/while/lstm_cell/ReadVariableOpReadVariableOp9sequential_lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Д
3sequential/lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Ж
5sequential/lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   Ж
5sequential/lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Л
-sequential/lstm/while/lstm_cell/strided_sliceStridedSlice6sequential/lstm/while/lstm_cell/ReadVariableOp:value:0<sequential/lstm/while/lstm_cell/strided_slice/stack:output:0>sequential/lstm/while/lstm_cell/strided_slice/stack_1:output:0>sequential/lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask≈
(sequential/lstm/while/lstm_cell/MatMul_4MatMul'sequential/lstm/while/lstm_cell/mul:z:06sequential/lstm/while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@ƒ
#sequential/lstm/while/lstm_cell/addAddV20sequential/lstm/while/lstm_cell/BiasAdd:output:02sequential/lstm/while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@Н
'sequential/lstm/while/lstm_cell/SigmoidSigmoid'sequential/lstm/while/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ђ
0sequential/lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp9sequential_lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Ж
5sequential/lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   И
7sequential/lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   И
7sequential/lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
/sequential/lstm/while/lstm_cell/strided_slice_1StridedSlice8sequential/lstm/while/lstm_cell/ReadVariableOp_1:value:0>sequential/lstm/while/lstm_cell/strided_slice_1/stack:output:0@sequential/lstm/while/lstm_cell/strided_slice_1/stack_1:output:0@sequential/lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask…
(sequential/lstm/while/lstm_cell/MatMul_5MatMul)sequential/lstm/while/lstm_cell/mul_1:z:08sequential/lstm/while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@»
%sequential/lstm/while/lstm_cell/add_1AddV22sequential/lstm/while/lstm_cell/BiasAdd_1:output:02sequential/lstm/while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@С
)sequential/lstm/while/lstm_cell/Sigmoid_1Sigmoid)sequential/lstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@≤
%sequential/lstm/while/lstm_cell/mul_4Mul-sequential/lstm/while/lstm_cell/Sigmoid_1:y:0#sequential_lstm_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@Ђ
0sequential/lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp9sequential_lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Ж
5sequential/lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   И
7sequential/lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   И
7sequential/lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
/sequential/lstm/while/lstm_cell/strided_slice_2StridedSlice8sequential/lstm/while/lstm_cell/ReadVariableOp_2:value:0>sequential/lstm/while/lstm_cell/strided_slice_2/stack:output:0@sequential/lstm/while/lstm_cell/strided_slice_2/stack_1:output:0@sequential/lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask…
(sequential/lstm/while/lstm_cell/MatMul_6MatMul)sequential/lstm/while/lstm_cell/mul_2:z:08sequential/lstm/while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@»
%sequential/lstm/while/lstm_cell/add_2AddV22sequential/lstm/while/lstm_cell/BiasAdd_2:output:02sequential/lstm/while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@Й
$sequential/lstm/while/lstm_cell/ReluRelu)sequential/lstm/while/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@њ
%sequential/lstm/while/lstm_cell/mul_5Mul+sequential/lstm/while/lstm_cell/Sigmoid:y:02sequential/lstm/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@ґ
%sequential/lstm/while/lstm_cell/add_3AddV2)sequential/lstm/while/lstm_cell/mul_4:z:0)sequential/lstm/while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ђ
0sequential/lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp9sequential_lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0Ж
5sequential/lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   И
7sequential/lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        И
7sequential/lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
/sequential/lstm/while/lstm_cell/strided_slice_3StridedSlice8sequential/lstm/while/lstm_cell/ReadVariableOp_3:value:0>sequential/lstm/while/lstm_cell/strided_slice_3/stack:output:0@sequential/lstm/while/lstm_cell/strided_slice_3/stack_1:output:0@sequential/lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask…
(sequential/lstm/while/lstm_cell/MatMul_7MatMul)sequential/lstm/while/lstm_cell/mul_3:z:08sequential/lstm/while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@»
%sequential/lstm/while/lstm_cell/add_4AddV22sequential/lstm/while/lstm_cell/BiasAdd_3:output:02sequential/lstm/while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@С
)sequential/lstm/while/lstm_cell/Sigmoid_2Sigmoid)sequential/lstm/while/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
&sequential/lstm/while/lstm_cell/Relu_1Relu)sequential/lstm/while/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@√
%sequential/lstm/while/lstm_cell/mul_6Mul-sequential/lstm/while/lstm_cell/Sigmoid_2:y:04sequential/lstm/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@В
:sequential/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#sequential_lstm_while_placeholder_1!sequential_lstm_while_placeholder)sequential/lstm/while/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype0:йи“]
sequential/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :М
sequential/lstm/while/addAddV2!sequential_lstm_while_placeholder$sequential/lstm/while/add/y:output:0*
T0*
_output_shapes
: _
sequential/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :І
sequential/lstm/while/add_1AddV28sequential_lstm_while_sequential_lstm_while_loop_counter&sequential/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: Й
sequential/lstm/while/IdentityIdentitysequential/lstm/while/add_1:z:0^sequential/lstm/while/NoOp*
T0*
_output_shapes
: ™
 sequential/lstm/while/Identity_1Identity>sequential_lstm_while_sequential_lstm_while_maximum_iterations^sequential/lstm/while/NoOp*
T0*
_output_shapes
: Й
 sequential/lstm/while/Identity_2Identitysequential/lstm/while/add:z:0^sequential/lstm/while/NoOp*
T0*
_output_shapes
: ґ
 sequential/lstm/while/Identity_3IdentityJsequential/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential/lstm/while/NoOp*
T0*
_output_shapes
: ¶
 sequential/lstm/while/Identity_4Identity)sequential/lstm/while/lstm_cell/mul_6:z:0^sequential/lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@¶
 sequential/lstm/while/Identity_5Identity)sequential/lstm/while/lstm_cell/add_3:z:0^sequential/lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Ц
sequential/lstm/while/NoOpNoOp/^sequential/lstm/while/lstm_cell/ReadVariableOp1^sequential/lstm/while/lstm_cell/ReadVariableOp_11^sequential/lstm/while/lstm_cell/ReadVariableOp_21^sequential/lstm/while/lstm_cell/ReadVariableOp_35^sequential/lstm/while/lstm_cell/split/ReadVariableOp7^sequential/lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "I
sequential_lstm_while_identity'sequential/lstm/while/Identity:output:0"M
 sequential_lstm_while_identity_1)sequential/lstm/while/Identity_1:output:0"M
 sequential_lstm_while_identity_2)sequential/lstm/while/Identity_2:output:0"M
 sequential_lstm_while_identity_3)sequential/lstm/while/Identity_3:output:0"M
 sequential_lstm_while_identity_4)sequential/lstm/while/Identity_4:output:0"M
 sequential_lstm_while_identity_5)sequential/lstm/while/Identity_5:output:0"t
7sequential_lstm_while_lstm_cell_readvariableop_resource9sequential_lstm_while_lstm_cell_readvariableop_resource_0"Д
?sequential_lstm_while_lstm_cell_split_1_readvariableop_resourceAsequential_lstm_while_lstm_cell_split_1_readvariableop_resource_0"А
=sequential_lstm_while_lstm_cell_split_readvariableop_resource?sequential_lstm_while_lstm_cell_split_readvariableop_resource_0"p
5sequential_lstm_while_sequential_lstm_strided_slice_17sequential_lstm_while_sequential_lstm_strided_slice_1_0"и
qsequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensorssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2`
.sequential/lstm/while/lstm_cell/ReadVariableOp.sequential/lstm/while/lstm_cell/ReadVariableOp2d
0sequential/lstm/while/lstm_cell/ReadVariableOp_10sequential/lstm/while/lstm_cell/ReadVariableOp_12d
0sequential/lstm/while/lstm_cell/ReadVariableOp_20sequential/lstm/while/lstm_cell/ReadVariableOp_22d
0sequential/lstm/while/lstm_cell/ReadVariableOp_30sequential/lstm/while/lstm_cell/ReadVariableOp_32l
4sequential/lstm/while/lstm_cell/split/ReadVariableOp4sequential/lstm/while/lstm_cell/split/ReadVariableOp2p
6sequential/lstm/while/lstm_cell/split_1/ReadVariableOp6sequential/lstm/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
оi
В	
while_body_348186
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
АА@
1while_lstm_cell_split_1_readvariableop_resource_0:	А<
)while_lstm_cell_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
АА>
/while_lstm_cell_split_1_readvariableop_resource:	А:
'while_lstm_cell_readvariableop_resource:	@АИҐwhile/lstm_cell/ReadVariableOpҐ while/lstm_cell/ReadVariableOp_1Ґ while/lstm_cell/ReadVariableOp_2Ґ while/lstm_cell/ReadVariableOp_3Ґ$while/lstm_cell/split/ReadVariableOpҐ&while/lstm_cell/split_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€А*
element_dtype0b
while/lstm_cell/ones_like/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:d
while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?І
while/lstm_cell/ones_likeFill(while/lstm_cell/ones_like/Shape:output:0(while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0“
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_split§
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@¶
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Х
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0ƒ
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitШ
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@Ь
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@Е
while/lstm_cell/mulMulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
while/lstm_cell/mul_1Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
while/lstm_cell/mul_2Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
while/lstm_cell/mul_3Mulwhile_placeholder_2"while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Й
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ї
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskХ
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul:z:0&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ф
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@m
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_1:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@В
while/lstm_cell/mul_4Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_2:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@i
while/lstm_cell/ReluReluwhile/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@П
while/lstm_cell/mul_5Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
while/lstm_cell/add_3AddV2while/lstm_cell/mul_4:z:0while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Л
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ≈
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЩ
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_3:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@q
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@У
while/lstm_cell/mul_6Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@¬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_6:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@v
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@¶

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
у

°
+__inference_sequential_layer_call_fn_347007

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: 
	unknown_3:
АА
	unknown_4:	А
	unknown_5:	@А
	unknown_6:@ 
	unknown_7: 
	unknown_8: 
	unknown_9:
identityИҐStatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*-
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_346276o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:€€€€€€€€€: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
З-
ѓ
F__inference_sequential_layer_call_and_return_conditional_losses_346811

inputs1
time_distributed_346773:%
time_distributed_346775:3
time_distributed_1_346780: '
time_distributed_1_346782: 
lstm_346793:
АА
lstm_346795:	А
lstm_346797:	@А
dense_346800:@ 
dense_346802:  
dense_1_346805: 
dense_1_346807:
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐlstm/StatefulPartitionedCallҐ(time_distributed/StatefulPartitionedCallҐ*time_distributed_1/StatefulPartitionedCall°
(time_distributed/StatefulPartitionedCallStatefulPartitionedCallinputstime_distributed_346773time_distributed_346775*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_346734w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         О
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€‘
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_346780time_distributed_1_346782*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€	 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_346701y
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         љ
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€К
"time_distributed_2/PartitionedCallPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_346672y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          њ
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 ы
"time_distributed_3/PartitionedCallPartitionedCall+time_distributed_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_346653y
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          Ј
time_distributed_3/ReshapeReshape+time_distributed_2/PartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ Щ
lstm/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0lstm_346793lstm_346795lstm_346797*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_346627И
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_346800dense_346802*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_346253С
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_346805dense_1_346807*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_346269w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€€
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:€€€€€€€€€: : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
№о
в

F__inference_sequential_layer_call_and_return_conditional_losses_347664

inputsP
6time_distributed_conv2d_conv2d_readvariableop_resource:E
7time_distributed_conv2d_biasadd_readvariableop_resource:T
:time_distributed_1_conv2d_1_conv2d_readvariableop_resource: I
;time_distributed_1_conv2d_1_biasadd_readvariableop_resource: @
,lstm_lstm_cell_split_readvariableop_resource:
АА=
.lstm_lstm_cell_split_1_readvariableop_resource:	А9
&lstm_lstm_cell_readvariableop_resource:	@А6
$dense_matmul_readvariableop_resource:@ 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐlstm/lstm_cell/ReadVariableOpҐlstm/lstm_cell/ReadVariableOp_1Ґlstm/lstm_cell/ReadVariableOp_2Ґlstm/lstm_cell/ReadVariableOp_3Ґ#lstm/lstm_cell/split/ReadVariableOpҐ%lstm/lstm_cell/split_1/ReadVariableOpҐ
lstm/whileҐ.time_distributed/conv2d/BiasAdd/ReadVariableOpҐ-time_distributed/conv2d/Conv2D/ReadVariableOpҐ2time_distributed_1/conv2d_1/BiasAdd/ReadVariableOpҐ1time_distributed_1/conv2d_1/Conv2D/ReadVariableOpw
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         О
time_distributed/ReshapeReshapeinputs'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ђ
-time_distributed/conv2d/Conv2D/ReadVariableOpReadVariableOp6time_distributed_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0е
time_distributed/conv2d/Conv2DConv2D!time_distributed/Reshape:output:05time_distributed/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
Ґ
.time_distributed/conv2d/BiasAdd/ReadVariableOpReadVariableOp7time_distributed_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0≈
time_distributed/conv2d/BiasAddBiasAdd'time_distributed/conv2d/Conv2D:output:06time_distributed/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€И
time_distributed/conv2d/ReluRelu(time_distributed/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€}
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€            Ї
time_distributed/Reshape_1Reshape*time_distributed/conv2d/Relu:activations:0)time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€y
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         Т
time_distributed/Reshape_2Reshapeinputs)time_distributed/Reshape_2/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€y
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ѓ
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€і
1time_distributed_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp:time_distributed_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0п
"time_distributed_1/conv2d_1/Conv2DConv2D#time_distributed_1/Reshape:output:09time_distributed_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 *
paddingVALID*
strides
™
2time_distributed_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp;time_distributed_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0—
#time_distributed_1/conv2d_1/BiasAddBiasAdd+time_distributed_1/conv2d_1/Conv2D:output:0:time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 Р
 time_distributed_1/conv2d_1/ReluRelu,time_distributed_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€   	          ¬
time_distributed_1/Reshape_1Reshape.time_distributed_1/conv2d_1/Relu:activations:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€	 {
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         ≥
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          ±
time_distributed_2/ReshapeReshape%time_distributed_1/Reshape_1:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 ≈
(time_distributed_2/max_pooling2d/MaxPoolMaxPool#time_distributed_2/Reshape:output:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides

"time_distributed_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€             ≈
time_distributed_2/Reshape_1Reshape1time_distributed_2/max_pooling2d/MaxPool:output:0+time_distributed_2/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€ {
"time_distributed_2/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          µ
time_distributed_2/Reshape_2Reshape%time_distributed_1/Reshape_1:output:0+time_distributed_2/Reshape_2/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 y
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          ±
time_distributed_3/ReshapeReshape%time_distributed_2/Reshape_1:output:0)time_distributed_3/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ q
 time_distributed_3/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ∞
"time_distributed_3/flatten/ReshapeReshape#time_distributed_3/Reshape:output:0)time_distributed_3/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аw
"time_distributed_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"€€€€      Є
time_distributed_3/Reshape_1Reshape+time_distributed_3/flatten/Reshape:output:0+time_distributed_3/Reshape_1/shape:output:0*
T0*,
_output_shapes
:€€€€€€€€€А{
"time_distributed_3/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          µ
time_distributed_3/Reshape_2Reshape%time_distributed_2/Reshape_1:output:0+time_distributed_3/Reshape_2/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ _

lstm/ShapeShape%time_distributed_3/Reshape_1:output:0*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@В
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:U
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@W
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@Ж
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Б
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@h
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
lstm/transpose	Transpose%time_distributed_3/Reshape_1:output:0lstm/transpose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€АN
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:d
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€√
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Л
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   п
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“d
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_maska
lstm/lstm_cell/ones_like/ShapeShapelstm/zeros:output:0*
T0*
_output_shapes
:c
lstm/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?§
lstm/lstm_cell/ones_likeFill'lstm/lstm_cell/ones_like/Shape:output:0'lstm/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
lstm/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Э
lstm/lstm_cell/dropout/MulMul!lstm/lstm_cell/ones_like:output:0%lstm/lstm_cell/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@m
lstm/lstm_cell/dropout/ShapeShape!lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:…
3lstm/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform%lstm/lstm_cell/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ѕЪЊj
%lstm/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?”
#lstm/lstm_cell/dropout/GreaterEqualGreaterEqual<lstm/lstm_cell/dropout/random_uniform/RandomUniform:output:0.lstm/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Н
lstm/lstm_cell/dropout/CastCast'lstm/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Ц
lstm/lstm_cell/dropout/Mul_1Mullstm/lstm_cell/dropout/Mul:z:0lstm/lstm_cell/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@c
lstm/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @°
lstm/lstm_cell/dropout_1/MulMul!lstm/lstm_cell/ones_like:output:0'lstm/lstm_cell/dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@o
lstm/lstm_cell/dropout_1/ShapeShape!lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:Ќ
5lstm/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2—сљl
'lstm/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ў
%lstm/lstm_cell/dropout_1/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_1/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@С
lstm/lstm_cell/dropout_1/CastCast)lstm/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Ь
lstm/lstm_cell/dropout_1/Mul_1Mul lstm/lstm_cell/dropout_1/Mul:z:0!lstm/lstm_cell/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@c
lstm/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @°
lstm/lstm_cell/dropout_2/MulMul!lstm/lstm_cell/ones_like:output:0'lstm/lstm_cell/dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@o
lstm/lstm_cell/dropout_2/ShapeShape!lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:Ќ
5lstm/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2…ьТl
'lstm/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ў
%lstm/lstm_cell/dropout_2/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_2/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@С
lstm/lstm_cell/dropout_2/CastCast)lstm/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Ь
lstm/lstm_cell/dropout_2/Mul_1Mul lstm/lstm_cell/dropout_2/Mul:z:0!lstm/lstm_cell/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@c
lstm/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @°
lstm/lstm_cell/dropout_3/MulMul!lstm/lstm_cell/ones_like:output:0'lstm/lstm_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@o
lstm/lstm_cell/dropout_3/ShapeShape!lstm/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:Ќ
5lstm/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform'lstm/lstm_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2шѓшl
'lstm/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ў
%lstm/lstm_cell/dropout_3/GreaterEqualGreaterEqual>lstm/lstm_cell/dropout_3/random_uniform/RandomUniform:output:00lstm/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@С
lstm/lstm_cell/dropout_3/CastCast)lstm/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@Ь
lstm/lstm_cell/dropout_3/Mul_1Mul lstm/lstm_cell/dropout_3/Mul:z:0!lstm/lstm_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@`
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
#lstm/lstm_cell/split/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
АА*
dtype0ѕ
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0+lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitП
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@С
lstm/lstm_cell/MatMul_1MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@С
lstm/lstm_cell/MatMul_2MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@С
lstm/lstm_cell/MatMul_3MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@b
 lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : С
%lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Ѕ
lstm/lstm_cell/split_1Split)lstm/lstm_cell/split_1/split_dim:output:0-lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitХ
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/MatMul:product:0lstm/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Щ
lstm/lstm_cell/BiasAdd_1BiasAdd!lstm/lstm_cell/MatMul_1:product:0lstm/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@Щ
lstm/lstm_cell/BiasAdd_2BiasAdd!lstm/lstm_cell/MatMul_2:product:0lstm/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@Щ
lstm/lstm_cell/BiasAdd_3BiasAdd!lstm/lstm_cell/MatMul_3:product:0lstm/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm/lstm_cell/mulMullstm/zeros:output:0 lstm/lstm_cell/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm/lstm_cell/mul_1Mullstm/zeros:output:0"lstm/lstm_cell/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm/lstm_cell/mul_2Mullstm/zeros:output:0"lstm/lstm_cell/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ж
lstm/lstm_cell/mul_3Mullstm/zeros:output:0"lstm/lstm_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Е
lstm/lstm_cell/ReadVariableOpReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0s
"lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   u
$lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ґ
lstm/lstm_cell/strided_sliceStridedSlice%lstm/lstm_cell/ReadVariableOp:value:0+lstm/lstm_cell/strided_slice/stack:output:0-lstm/lstm_cell/strided_slice/stack_1:output:0-lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskТ
lstm/lstm_cell/MatMul_4MatMullstm/lstm_cell/mul:z:0%lstm/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@С
lstm/lstm_cell/addAddV2lstm/lstm_cell/BiasAdd:output:0!lstm/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@k
lstm/lstm_cell/SigmoidSigmoidlstm/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@З
lstm/lstm_cell/ReadVariableOp_1ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0u
$lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   w
&lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   w
&lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
lstm/lstm_cell/strided_slice_1StridedSlice'lstm/lstm_cell/ReadVariableOp_1:value:0-lstm/lstm_cell/strided_slice_1/stack:output:0/lstm/lstm_cell/strided_slice_1/stack_1:output:0/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЦ
lstm/lstm_cell/MatMul_5MatMullstm/lstm_cell/mul_1:z:0'lstm/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Х
lstm/lstm_cell/add_1AddV2!lstm/lstm_cell/BiasAdd_1:output:0!lstm/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@o
lstm/lstm_cell/Sigmoid_1Sigmoidlstm/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@В
lstm/lstm_cell/mul_4Mullstm/lstm_cell/Sigmoid_1:y:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@З
lstm/lstm_cell/ReadVariableOp_2ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0u
$lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   w
&lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   w
&lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
lstm/lstm_cell/strided_slice_2StridedSlice'lstm/lstm_cell/ReadVariableOp_2:value:0-lstm/lstm_cell/strided_slice_2/stack:output:0/lstm/lstm_cell/strided_slice_2/stack_1:output:0/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЦ
lstm/lstm_cell/MatMul_6MatMullstm/lstm_cell/mul_2:z:0'lstm/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Х
lstm/lstm_cell/add_2AddV2!lstm/lstm_cell/BiasAdd_2:output:0!lstm/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@g
lstm/lstm_cell/ReluRelulstm/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@М
lstm/lstm_cell/mul_5Mullstm/lstm_cell/Sigmoid:y:0!lstm/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@Г
lstm/lstm_cell/add_3AddV2lstm/lstm_cell/mul_4:z:0lstm/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@З
lstm/lstm_cell/ReadVariableOp_3ReadVariableOp&lstm_lstm_cell_readvariableop_resource*
_output_shapes
:	@А*
dtype0u
$lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   w
&lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
lstm/lstm_cell/strided_slice_3StridedSlice'lstm/lstm_cell/ReadVariableOp_3:value:0-lstm/lstm_cell/strided_slice_3/stack:output:0/lstm/lstm_cell/strided_slice_3/stack_1:output:0/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maskЦ
lstm/lstm_cell/MatMul_7MatMullstm/lstm_cell/mul_3:z:0'lstm/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Х
lstm/lstm_cell/add_4AddV2!lstm/lstm_cell/BiasAdd_3:output:0!lstm/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@o
lstm/lstm_cell/Sigmoid_2Sigmoidlstm/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@i
lstm/lstm_cell/Relu_1Relulstm/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Р
lstm/lstm_cell/mul_6Mullstm/lstm_cell/Sigmoid_2:y:0#lstm/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@s
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   «
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“K
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : h
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Y
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : µ

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_lstm_cell_split_readvariableop_resource.lstm_lstm_cell_split_1_readvariableop_resource&lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *"
bodyR
lstm_while_body_347492*"
condR
lstm_while_cond_347491*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations Ж
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   —
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype0m
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€f
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:†
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_maskj
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          •
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@`
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0М
dense/MatMulMatMullstm/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Л
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€п
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^lstm/lstm_cell/ReadVariableOp ^lstm/lstm_cell/ReadVariableOp_1 ^lstm/lstm_cell/ReadVariableOp_2 ^lstm/lstm_cell/ReadVariableOp_3$^lstm/lstm_cell/split/ReadVariableOp&^lstm/lstm_cell/split_1/ReadVariableOp^lstm/while/^time_distributed/conv2d/BiasAdd/ReadVariableOp.^time_distributed/conv2d/Conv2D/ReadVariableOp3^time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp2^time_distributed_1/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:€€€€€€€€€: : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2>
lstm/lstm_cell/ReadVariableOplstm/lstm_cell/ReadVariableOp2B
lstm/lstm_cell/ReadVariableOp_1lstm/lstm_cell/ReadVariableOp_12B
lstm/lstm_cell/ReadVariableOp_2lstm/lstm_cell/ReadVariableOp_22B
lstm/lstm_cell/ReadVariableOp_3lstm/lstm_cell/ReadVariableOp_32J
#lstm/lstm_cell/split/ReadVariableOp#lstm/lstm_cell/split/ReadVariableOp2N
%lstm/lstm_cell/split_1/ReadVariableOp%lstm/lstm_cell/split_1/ReadVariableOp2

lstm/while
lstm/while2`
.time_distributed/conv2d/BiasAdd/ReadVariableOp.time_distributed/conv2d/BiasAdd/ReadVariableOp2^
-time_distributed/conv2d/Conv2D/ReadVariableOp-time_distributed/conv2d/Conv2D/ReadVariableOp2h
2time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp2time_distributed_1/conv2d_1/BiasAdd/ReadVariableOp2f
1time_distributed_1/conv2d_1/Conv2D/ReadVariableOp1time_distributed_1/conv2d_1/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
µ
Ђ
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_347877

inputsA
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: 
identityИҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€О
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ґ
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 *
paddingVALID*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 l
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€   	          Й
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€	 m
IdentityIdentityReshape_1:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€	 Й
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
«-
њ
F__inference_sequential_layer_call_and_return_conditional_losses_346904
time_distributed_input1
time_distributed_346866:%
time_distributed_346868:3
time_distributed_1_346873: '
time_distributed_1_346875: 
lstm_346886:
АА
lstm_346888:	А
lstm_346890:	@А
dense_346893:@ 
dense_346895:  
dense_1_346898: 
dense_1_346900:
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐlstm/StatefulPartitionedCallҐ(time_distributed/StatefulPartitionedCallҐ*time_distributed_1/StatefulPartitionedCall±
(time_distributed/StatefulPartitionedCallStatefulPartitionedCalltime_distributed_inputtime_distributed_346866time_distributed_346868*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_345951w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         Ю
time_distributed/ReshapeReshapetime_distributed_input'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€‘
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_346873time_distributed_1_346875*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€	 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_345973y
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         љ
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€К
"time_distributed_2/PartitionedCallPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_345989y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          њ
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 ы
"time_distributed_3/PartitionedCallPartitionedCall+time_distributed_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_346002y
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          Ј
time_distributed_3/ReshapeReshape+time_distributed_2/PartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ Щ
lstm/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0lstm_346886lstm_346888lstm_346890*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_346234И
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_346893dense_346895*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_346253С
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_346898dense_1_346900*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_346269w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€€
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:€€€€€€€€€: : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:k g
3
_output_shapes!
:€€€€€€€€€
0
_user_specified_nametime_distributed_input
н
ф
*__inference_lstm_cell_layer_call_fn_349272

inputs
states_0
states_1
unknown:
АА
	unknown_0:	А
	unknown_1:	@А
identity

identity_1

identity_2ИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_345793o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€А:€€€€€€€€€@:€€€€€€€€€@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/1
С
µ
%__inference_lstm_layer_call_fn_348051
inputs_0
unknown:
АА
	unknown_0:	А
	unknown_1:	@А
identityИҐStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_345655o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А
"
_user_specified_name
inputs/0
≈
_
C__inference_flatten_layer_call_and_return_conditional_losses_349238

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€ :W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ў
Џ
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_345328

inputs)
conv2d_1_345316: 
conv2d_1_345318: 
identityИҐ conv2d_1/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€З
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0conv2d_1_345316conv2d_1_345318*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€	 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_345274\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : Ќ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:†
	Reshape_1Reshape)conv2d_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 i
NoOpNoOp!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&€€€€€€€€€€€€€€€€€€: : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
•
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_345365

inputs
identityЗ
MaxPoolMaxPoolinputs*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€	 :W S
/
_output_shapes
:€€€€€€€€€	 
 
_user_specified_nameinputs
£
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_346672

inputs
identityf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 Я
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
l
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€             М
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€ f
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€	 :[ W
3
_output_shapes!
:€€€€€€€€€	 
 
_user_specified_nameinputs
ж
O
3__inference_time_distributed_2_layer_call_fn_347907

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_345989l
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€	 :[ W
3
_output_shapes!
:€€€€€€€€€	 
 
_user_specified_nameinputs
у

°
+__inference_sequential_layer_call_fn_347034

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: 
	unknown_3:
АА
	unknown_4:	А
	unknown_5:	@А
	unknown_6:@ 
	unknown_7: 
	unknown_8: 
	unknown_9:
identityИҐStatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*-
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_346811o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:€€€€€€€€€: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
У
®
3__inference_time_distributed_1_layer_call_fn_347814

inputs!
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€	 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_346701{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€	 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
В
ы
B__inference_conv2d_layer_call_and_return_conditional_losses_349187

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ъs
±

lstm_while_body_347177&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0H
4lstm_while_lstm_cell_split_readvariableop_resource_0:
ААE
6lstm_while_lstm_cell_split_1_readvariableop_resource_0:	АA
.lstm_while_lstm_cell_readvariableop_resource_0:	@А
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorF
2lstm_while_lstm_cell_split_readvariableop_resource:
ААC
4lstm_while_lstm_cell_split_1_readvariableop_resource:	А?
,lstm_while_lstm_cell_readvariableop_resource:	@АИҐ#lstm/while/lstm_cell/ReadVariableOpҐ%lstm/while/lstm_cell/ReadVariableOp_1Ґ%lstm/while/lstm_cell/ReadVariableOp_2Ґ%lstm/while/lstm_cell/ReadVariableOp_3Ґ)lstm/while/lstm_cell/split/ReadVariableOpҐ+lstm/while/lstm_cell/split_1/ReadVariableOpН
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ј
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€А*
element_dtype0l
$lstm/while/lstm_cell/ones_like/ShapeShapelstm_while_placeholder_2*
T0*
_output_shapes
:i
$lstm/while/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?ґ
lstm/while/lstm_cell/ones_likeFill-lstm/while/lstm_cell/ones_like/Shape:output:0-lstm/while/lstm_cell/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@f
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :†
)lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp4lstm_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
АА*
dtype0б
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:01lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_split≥
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@µ
lstm/while/lstm_cell/MatMul_1MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@µ
lstm/while/lstm_cell/MatMul_2MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@µ
lstm/while/lstm_cell/MatMul_3MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@h
&lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Я
+lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype0”
lstm/while/lstm_cell/split_1Split/lstm/while/lstm_cell/split_1/split_dim:output:03lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splitІ
lstm/while/lstm_cell/BiasAddBiasAdd%lstm/while/lstm_cell/MatMul:product:0%lstm/while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ђ
lstm/while/lstm_cell/BiasAdd_1BiasAdd'lstm/while/lstm_cell/MatMul_1:product:0%lstm/while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@Ђ
lstm/while/lstm_cell/BiasAdd_2BiasAdd'lstm/while/lstm_cell/MatMul_2:product:0%lstm/while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@Ђ
lstm/while/lstm_cell/BiasAdd_3BiasAdd'lstm/while/lstm_cell/MatMul_3:product:0%lstm/while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@Ф
lstm/while/lstm_cell/mulMullstm_while_placeholder_2'lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ц
lstm/while/lstm_cell/mul_1Mullstm_while_placeholder_2'lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ц
lstm/while/lstm_cell/mul_2Mullstm_while_placeholder_2'lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ц
lstm/while/lstm_cell/mul_3Mullstm_while_placeholder_2'lstm/while/lstm_cell/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@У
#lstm/while/lstm_cell/ReadVariableOpReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0y
(lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   {
*lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ‘
"lstm/while/lstm_cell/strided_sliceStridedSlice+lstm/while/lstm_cell/ReadVariableOp:value:01lstm/while/lstm_cell/strided_slice/stack:output:03lstm/while/lstm_cell/strided_slice/stack_1:output:03lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask§
lstm/while/lstm_cell/MatMul_4MatMullstm/while/lstm_cell/mul:z:0+lstm/while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@£
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/BiasAdd:output:0'lstm/while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@w
lstm/while/lstm_cell/SigmoidSigmoidlstm/while/lstm_cell/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Х
%lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0{
*lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   }
,lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   }
,lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ё
$lstm/while/lstm_cell/strided_slice_1StridedSlice-lstm/while/lstm_cell/ReadVariableOp_1:value:03lstm/while/lstm_cell/strided_slice_1/stack:output:05lstm/while/lstm_cell/strided_slice_1/stack_1:output:05lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask®
lstm/while/lstm_cell/MatMul_5MatMullstm/while/lstm_cell/mul_1:z:0-lstm/while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@І
lstm/while/lstm_cell/add_1AddV2'lstm/while/lstm_cell/BiasAdd_1:output:0'lstm/while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@{
lstm/while/lstm_cell/Sigmoid_1Sigmoidlstm/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@С
lstm/while/lstm_cell/mul_4Mul"lstm/while/lstm_cell/Sigmoid_1:y:0lstm_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@Х
%lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0{
*lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   }
,lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   }
,lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ё
$lstm/while/lstm_cell/strided_slice_2StridedSlice-lstm/while/lstm_cell/ReadVariableOp_2:value:03lstm/while/lstm_cell/strided_slice_2/stack:output:05lstm/while/lstm_cell/strided_slice_2/stack_1:output:05lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask®
lstm/while/lstm_cell/MatMul_6MatMullstm/while/lstm_cell/mul_2:z:0-lstm/while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@І
lstm/while/lstm_cell/add_2AddV2'lstm/while/lstm_cell/BiasAdd_2:output:0'lstm/while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@s
lstm/while/lstm_cell/ReluRelulstm/while/lstm_cell/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ю
lstm/while/lstm_cell/mul_5Mul lstm/while/lstm_cell/Sigmoid:y:0'lstm/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@Х
lstm/while/lstm_cell/add_3AddV2lstm/while/lstm_cell/mul_4:z:0lstm/while/lstm_cell/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Х
%lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	@А*
dtype0{
*lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   }
,lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ё
$lstm/while/lstm_cell/strided_slice_3StridedSlice-lstm/while/lstm_cell/ReadVariableOp_3:value:03lstm/while/lstm_cell/strided_slice_3/stack:output:05lstm/while/lstm_cell/strided_slice_3/stack_1:output:05lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask®
lstm/while/lstm_cell/MatMul_7MatMullstm/while/lstm_cell/mul_3:z:0-lstm/while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@І
lstm/while/lstm_cell/add_4AddV2'lstm/while/lstm_cell/BiasAdd_3:output:0'lstm/while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@{
lstm/while/lstm_cell/Sigmoid_2Sigmoidlstm/while/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@u
lstm/while/lstm_cell/Relu_1Relulstm/while/lstm_cell/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@Ґ
lstm/while/lstm_cell/mul_6Mul"lstm/while/lstm_cell/Sigmoid_2:y:0)lstm/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@÷
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_6:z:0*
_output_shapes
: *
element_dtype0:йи“R
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :k
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: T
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: h
lstm/while/IdentityIdentitylstm/while/add_1:z:0^lstm/while/NoOp*
T0*
_output_shapes
: ~
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations^lstm/while/NoOp*
T0*
_output_shapes
: h
lstm/while/Identity_2Identitylstm/while/add:z:0^lstm/while/NoOp*
T0*
_output_shapes
: Х
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm/while/NoOp*
T0*
_output_shapes
: Е
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_6:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Е
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_3:z:0^lstm/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@…
lstm/while/NoOpNoOp$^lstm/while/lstm_cell/ReadVariableOp&^lstm/while/lstm_cell/ReadVariableOp_1&^lstm/while/lstm_cell/ReadVariableOp_2&^lstm/while/lstm_cell/ReadVariableOp_3*^lstm/while/lstm_cell/split/ReadVariableOp,^lstm/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"^
,lstm_while_lstm_cell_readvariableop_resource.lstm_while_lstm_cell_readvariableop_resource_0"n
4lstm_while_lstm_cell_split_1_readvariableop_resource6lstm_while_lstm_cell_split_1_readvariableop_resource_0"j
2lstm_while_lstm_cell_split_readvariableop_resource4lstm_while_lstm_cell_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"Љ
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2J
#lstm/while/lstm_cell/ReadVariableOp#lstm/while/lstm_cell/ReadVariableOp2N
%lstm/while/lstm_cell/ReadVariableOp_1%lstm/while/lstm_cell/ReadVariableOp_12N
%lstm/while/lstm_cell/ReadVariableOp_2%lstm/while/lstm_cell/ReadVariableOp_22N
%lstm/while/lstm_cell/ReadVariableOp_3%lstm/while/lstm_cell/ReadVariableOp_32V
)lstm/while/lstm_cell/split/ReadVariableOp)lstm/while/lstm_cell/split/ReadVariableOp2Z
+lstm/while/lstm_cell/split_1/ReadVariableOp+lstm/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
В
ы
B__inference_conv2d_layer_call_and_return_conditional_losses_345188

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
•
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_349227

inputs
identityЗ
MaxPoolMaxPoolinputs*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€	 :W S
/
_output_shapes
:€€€€€€€€€	 
 
_user_specified_nameinputs
µ
√
while_cond_346467
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_346467___redundant_placeholder04
0while_while_cond_346467___redundant_placeholder14
0while_while_cond_346467___redundant_placeholder24
0while_while_cond_346467___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
µ
√
while_cond_348707
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_348707___redundant_placeholder04
0while_while_cond_348707___redundant_placeholder14
0while_while_cond_348707___redundant_placeholder24
0while_while_cond_348707___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
Ь
°
L__inference_time_distributed_layer_call_and_return_conditional_losses_347724

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identityИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€К
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0≤
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Т
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :Ќ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:Р
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€v
IdentityIdentityReshape_1:output:0^NoOp*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€Е
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&€€€€€€€€€€€€€€€€€€: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
£
±
+__inference_sequential_layer_call_fn_346863
time_distributed_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: 
	unknown_3:
АА
	unknown_4:	А
	unknown_5:	@А
	unknown_6:@ 
	unknown_7: 
	unknown_8: 
	unknown_9:
identityИҐStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCalltime_distributed_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*-
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_346811o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:€€€€€€€€€: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:k g
3
_output_shapes!
:€€€€€€€€€
0
_user_specified_nametime_distributed_input
£
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_347966

inputs
identityf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 Я
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
l
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€             М
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€ f
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€	 :[ W
3
_output_shapes!
:€€€€€€€€€	 
 
_user_specified_nameinputs
ш\
І
E__inference_lstm_cell_layer_call_and_return_conditional_losses_345793

inputs

states
states_11
split_readvariableop_resource:
АА.
split_1_readvariableop_resource:	А*
readvariableop_resource:	@А
identity

identity_1

identity_2ИҐReadVariableOpҐReadVariableOp_1ҐReadVariableOp_2ҐReadVariableOp_3Ґsplit/ReadVariableOpҐsplit_1/ReadVariableOpE
ones_like/ShapeShapestates*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:Ђ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2Ѕ√Ы[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¶
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:ѓ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2п∆В]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ђ
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@s
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@o
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:Ѓ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2АУ*]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ђ
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@s
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@o
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:Ѓ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2∆ђ@]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ђ
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@s
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@o
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ґ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	А@:	А@:	А@:	А@*
	num_splitZ
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€@\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€@\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€@\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€@S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype0Ф
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@W
mulMulstatesdropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@[
mul_1Mulstatesdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@[
mul_2Mulstatesdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@[
mul_3Mulstatesdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      л
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maske
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€@i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@W
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€@i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@_
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@V
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_maski
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@Q
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@K
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@c
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@X
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Z

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@ј
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€А:€€€€€€€€€@:€€€€€€€€€@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates
£
±
+__inference_sequential_layer_call_fn_346301
time_distributed_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: 
	unknown_3:
АА
	unknown_4:	А
	unknown_5:	@А
	unknown_6:@ 
	unknown_7: 
	unknown_8: 
	unknown_9:
identityИҐStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCalltime_distributed_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*-
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_346276o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:€€€€€€€€€: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:k g
3
_output_shapes!
:€€€€€€€€€
0
_user_specified_nametime_distributed_input
З
°
L__inference_time_distributed_layer_call_and_return_conditional_losses_346734

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identityИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€К
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0≤
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Т
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€l
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€            З
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€m
IdentityIdentityReshape_1:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€Е
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
∆	
ф
C__inference_dense_1_layer_call_and_return_conditional_losses_346269

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
У
®
3__inference_time_distributed_1_layer_call_fn_347805

inputs!
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€	 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_345973{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€	 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
Є
®
3__inference_time_distributed_1_layer_call_fn_347796

inputs!
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_345328Д
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:&€€€€€€€€€€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
З
°
L__inference_time_distributed_layer_call_and_return_conditional_losses_347778

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:
identityИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€К
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0≤
conv2d/Conv2DConv2DReshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Т
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€l
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€            З
	Reshape_1Reshapeconv2d/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€m
IdentityIdentityReshape_1:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€Е
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
µ
Ђ
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_345973

inputsA
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: 
identityИҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€О
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ґ
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 *
paddingVALID*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 l
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€   	          Й
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€	 m
IdentityIdentityReshape_1:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€	 Й
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs
щ
≥
%__inference_lstm_layer_call_fn_348084

inputs
unknown:
АА
	unknown_0:	А
	unknown_1:	@А
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_346627o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
щ
≥
%__inference_lstm_layer_call_fn_348073

inputs
unknown:
АА
	unknown_0:	А
	unknown_1:	@А
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_346234o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
«-
њ
F__inference_sequential_layer_call_and_return_conditional_losses_346945
time_distributed_input1
time_distributed_346907:%
time_distributed_346909:3
time_distributed_1_346914: '
time_distributed_1_346916: 
lstm_346927:
АА
lstm_346929:	А
lstm_346931:	@А
dense_346934:@ 
dense_346936:  
dense_1_346939: 
dense_1_346941:
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐlstm/StatefulPartitionedCallҐ(time_distributed/StatefulPartitionedCallҐ*time_distributed_1/StatefulPartitionedCall±
(time_distributed/StatefulPartitionedCallStatefulPartitionedCalltime_distributed_inputtime_distributed_346907time_distributed_346909*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_346734w
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         Ю
time_distributed/ReshapeReshapetime_distributed_input'time_distributed/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€‘
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0time_distributed_1_346914time_distributed_1_346916*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€	 *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_346701y
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         љ
time_distributed_1/ReshapeReshape1time_distributed/StatefulPartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€К
"time_distributed_2/PartitionedCallPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_346672y
 time_distributed_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          њ
time_distributed_2/ReshapeReshape3time_distributed_1/StatefulPartitionedCall:output:0)time_distributed_2/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 ы
"time_distributed_3/PartitionedCallPartitionedCall+time_distributed_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *W
fRRP
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_346653y
 time_distributed_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€          Ј
time_distributed_3/ReshapeReshape+time_distributed_2/PartitionedCall:output:0)time_distributed_3/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ Щ
lstm/StatefulPartitionedCallStatefulPartitionedCall+time_distributed_3/PartitionedCall:output:0lstm_346927lstm_346929lstm_346931*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_346627И
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_346934dense_346936*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_346253С
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_346939dense_1_346941*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_346269w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€€
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:€€€€€€€€€: : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:k g
3
_output_shapes!
:€€€€€€€€€
0
_user_specified_nametime_distributed_input
Г®
ы
"__inference__traced_restore_349739
file_prefix/
assignvariableop_dense_kernel:@ +
assignvariableop_1_dense_bias: 3
!assignvariableop_2_dense_1_kernel: -
assignvariableop_3_dense_1_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: D
*assignvariableop_9_time_distributed_kernel:7
)assignvariableop_10_time_distributed_bias:G
-assignvariableop_11_time_distributed_1_kernel: 9
+assignvariableop_12_time_distributed_1_bias: =
)assignvariableop_13_lstm_lstm_cell_kernel:
ААF
3assignvariableop_14_lstm_lstm_cell_recurrent_kernel:	@А6
'assignvariableop_15_lstm_lstm_cell_bias:	А#
assignvariableop_16_total: #
assignvariableop_17_count: %
assignvariableop_18_total_1: %
assignvariableop_19_count_1: 9
'assignvariableop_20_adam_dense_kernel_m:@ 3
%assignvariableop_21_adam_dense_bias_m: ;
)assignvariableop_22_adam_dense_1_kernel_m: 5
'assignvariableop_23_adam_dense_1_bias_m:L
2assignvariableop_24_adam_time_distributed_kernel_m:>
0assignvariableop_25_adam_time_distributed_bias_m:N
4assignvariableop_26_adam_time_distributed_1_kernel_m: @
2assignvariableop_27_adam_time_distributed_1_bias_m: D
0assignvariableop_28_adam_lstm_lstm_cell_kernel_m:
ААM
:assignvariableop_29_adam_lstm_lstm_cell_recurrent_kernel_m:	@А=
.assignvariableop_30_adam_lstm_lstm_cell_bias_m:	А9
'assignvariableop_31_adam_dense_kernel_v:@ 3
%assignvariableop_32_adam_dense_bias_v: ;
)assignvariableop_33_adam_dense_1_kernel_v: 5
'assignvariableop_34_adam_dense_1_bias_v:L
2assignvariableop_35_adam_time_distributed_kernel_v:>
0assignvariableop_36_adam_time_distributed_bias_v:N
4assignvariableop_37_adam_time_distributed_1_kernel_v: @
2assignvariableop_38_adam_time_distributed_1_bias_v: D
0assignvariableop_39_adam_lstm_lstm_cell_kernel_v:
ААM
:assignvariableop_40_adam_lstm_lstm_cell_recurrent_kernel_v:	@А=
.assignvariableop_41_adam_lstm_lstm_cell_bias_v:	А
identity_43ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9О
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*і
value™BІ+B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH∆
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ш
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¬
_output_shapesѓ
ђ:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_9AssignVariableOp*assignvariableop_9_time_distributed_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_10AssignVariableOp)assignvariableop_10_time_distributed_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_11AssignVariableOp-assignvariableop_11_time_distributed_1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_12AssignVariableOp+assignvariableop_12_time_distributed_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_13AssignVariableOp)assignvariableop_13_lstm_lstm_cell_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_14AssignVariableOp3assignvariableop_14_lstm_lstm_cell_recurrent_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_15AssignVariableOp'assignvariableop_15_lstm_lstm_cell_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_18AssignVariableOpassignvariableop_18_total_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_21AssignVariableOp%assignvariableop_21_adam_dense_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_1_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_1_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_time_distributed_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_time_distributed_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_time_distributed_1_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_time_distributed_1_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_28AssignVariableOp0assignvariableop_28_adam_lstm_lstm_cell_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_29AssignVariableOp:assignvariableop_29_adam_lstm_lstm_cell_recurrent_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_30AssignVariableOp.assignvariableop_30_adam_lstm_lstm_cell_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_dense_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_35AssignVariableOp2assignvariableop_35_adam_time_distributed_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_36AssignVariableOp0assignvariableop_36_adam_time_distributed_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_time_distributed_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adam_time_distributed_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_39AssignVariableOp0assignvariableop_39_adam_lstm_lstm_cell_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_40AssignVariableOp:assignvariableop_40_adam_lstm_lstm_cell_recurrent_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_41AssignVariableOp.assignvariableop_41_adam_lstm_lstm_cell_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 л
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_43IdentityIdentity_42:output:0^NoOp_1*
T0*
_output_shapes
: Ў
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_43Identity_43:output:0*i
_input_shapesX
V: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
и7
ч
@__inference_lstm_layer_call_and_return_conditional_losses_345655

inputs$
lstm_cell_345573:
АА
lstm_cell_345575:	А#
lstm_cell_345577:	@А
identityИҐ!lstm_cell/StatefulPartitionedCallҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€АD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€А*
shrink_axis_maskл
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_345573lstm_cell_345575lstm_cell_345577*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_345572n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_345573lstm_cell_345575lstm_cell_345577*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_345586*
condR
while_cond_345585*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@r
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€А: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
£
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_345989

inputs
identityf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 Я
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
l
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€             М
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€ f
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€	 :[ W
3
_output_shapes!
:€€€€€€€€€	 
 
_user_specified_nameinputs
Є
j
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_347948

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€	          l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 Я
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€S
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B : Ќ
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:Х
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ o
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:&€€€€€€€€€€€€€€€€€€	 :d `
<
_output_shapes*
(:&€€€€€€€€€€€€€€€€€€	 
 
_user_specified_nameinputs
µ
Ђ
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_346701

inputsA
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: 
identityИҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"€€€€         l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€О
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ґ
conv2d_1/Conv2DConv2DReshape:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 *
paddingVALID*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€	 j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€	 l
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"€€€€   	          Й
	Reshape_1Reshapeconv2d_1/Relu:activations:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:€€€€€€€€€	 m
IdentityIdentityReshape_1:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€	 Й
NoOpNoOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:[ W
3
_output_shapes!
:€€€€€€€€€
 
_user_specified_nameinputs"ВL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*‘
serving_defaultј
e
time_distributed_inputK
(serving_default_time_distributed_input:0€€€€€€€€€;
dense_10
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:аЭ
л
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	optimizer
	
signatures
#
_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
≈__call__
+∆&call_and_return_all_conditional_losses
«_default_save_signature"
_tf_keras_sequential
„
	layer
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
»__call__
+…&call_and_return_all_conditional_losses"
_tf_keras_layer
„
	layer
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+Ћ&call_and_return_all_conditional_losses"
_tf_keras_layer
„
	layer
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
 	keras_api
ћ__call__
+Ќ&call_and_return_all_conditional_losses"
_tf_keras_layer
„
	!layer
#"_self_saveable_object_factories
#	variables
$trainable_variables
%regularization_losses
&	keras_api
ќ__call__
+ѕ&call_and_return_all_conditional_losses"
_tf_keras_layer
к
'cell
(
state_spec
#)_self_saveable_object_factories
*	variables
+trainable_variables
,regularization_losses
-	keras_api
–__call__
+—&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
в

.kernel
/bias
#0_self_saveable_object_factories
1	variables
2trainable_variables
3regularization_losses
4	keras_api
“__call__
+”&call_and_return_all_conditional_losses"
_tf_keras_layer
в

5kernel
6bias
#7_self_saveable_object_factories
8	variables
9trainable_variables
:regularization_losses
;	keras_api
‘__call__
+’&call_and_return_all_conditional_losses"
_tf_keras_layer
ѓ
<iter

=beta_1

>beta_2
	?decay
@learning_rate.mѓ/m∞5m±6m≤Am≥BmіCmµDmґEmЈFmЄGmє.vЇ/vї5vЉ6vљAvЊBvњCvјDvЅEv¬Fv√Gvƒ"
	optimizer
-
÷serving_default"
signature_map
 "
trackable_dict_wrapper
n
A0
B1
C2
D3
E4
F5
G6
.7
/8
59
610"
trackable_list_wrapper
n
A0
B1
C2
D3
E4
F5
G6
.7
/8
59
610"
trackable_list_wrapper
 "
trackable_list_wrapper
ќ
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
≈__call__
«_default_save_signature
+∆&call_and_return_all_conditional_losses
'∆"call_and_return_conditional_losses"
_generic_user_object
в

Akernel
Bbias
#M_self_saveable_object_factories
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
„__call__
+Ў&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_dict_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
»__call__
+…&call_and_return_all_conditional_losses
'…"call_and_return_conditional_losses"
_generic_user_object
в

Ckernel
Dbias
#W_self_saveable_object_factories
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
ў__call__
+Џ&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_dict_wrapper
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
∞
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
ћ
#a_self_saveable_object_factories
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
џ__call__
+№&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
ћ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
ћ
#k_self_saveable_object_factories
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
Ё__call__
+ё&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
#	variables
$trainable_variables
%regularization_losses
ќ__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
И
u
state_size

Ekernel
Frecurrent_kernel
Gbias
#v_self_saveable_object_factories
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
я__call__
+а&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
E0
F1
G2"
trackable_list_wrapper
5
E0
F1
G2"
trackable_list_wrapper
 "
trackable_list_wrapper
љ

{states
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
Аlayer_metrics
*	variables
+trainable_variables
,regularization_losses
–__call__
+—&call_and_return_all_conditional_losses
'—"call_and_return_conditional_losses"
_generic_user_object
:@ 2dense/kernel
: 2
dense/bias
 "
trackable_dict_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
1	variables
2trainable_variables
3regularization_losses
“__call__
+”&call_and_return_all_conditional_losses
'”"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_1/kernel
:2dense_1/bias
 "
trackable_dict_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
8	variables
9trainable_variables
:regularization_losses
‘__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
1:/2time_distributed/kernel
#:!2time_distributed/bias
3:1 2time_distributed_1/kernel
%:# 2time_distributed_1/bias
):'
АА2lstm/lstm_cell/kernel
2:0	@А2lstm/lstm_cell/recurrent_kernel
": А2lstm/lstm_cell/bias
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
0
Л0
М1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
„__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
џ__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
†layer_metrics
l	variables
mtrainable_variables
nregularization_losses
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
!0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
E0
F1
G2"
trackable_list_wrapper
5
E0
F1
G2"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
°non_trainable_variables
Ґlayers
£metrics
 §layer_regularization_losses
•layer_metrics
w	variables
xtrainable_variables
yregularization_losses
я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
'0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

¶total

Іcount
®	variables
©	keras_api"
_tf_keras_metric
c

™total

Ђcount
ђ
_fn_kwargs
≠	variables
Ѓ	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
¶0
І1"
trackable_list_wrapper
.
®	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
™0
Ђ1"
trackable_list_wrapper
.
≠	variables"
_generic_user_object
#:!@ 2Adam/dense/kernel/m
: 2Adam/dense/bias/m
%:# 2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
6:42Adam/time_distributed/kernel/m
(:&2Adam/time_distributed/bias/m
8:6 2 Adam/time_distributed_1/kernel/m
*:( 2Adam/time_distributed_1/bias/m
.:,
АА2Adam/lstm/lstm_cell/kernel/m
7:5	@А2&Adam/lstm/lstm_cell/recurrent_kernel/m
':%А2Adam/lstm/lstm_cell/bias/m
#:!@ 2Adam/dense/kernel/v
: 2Adam/dense/bias/v
%:# 2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
6:42Adam/time_distributed/kernel/v
(:&2Adam/time_distributed/bias/v
8:6 2 Adam/time_distributed_1/kernel/v
*:( 2Adam/time_distributed_1/bias/v
.:,
АА2Adam/lstm/lstm_cell/kernel/v
7:5	@А2&Adam/lstm/lstm_cell/recurrent_kernel/v
':%А2Adam/lstm/lstm_cell/bias/v
ъ2ч
+__inference_sequential_layer_call_fn_346301
+__inference_sequential_layer_call_fn_347007
+__inference_sequential_layer_call_fn_347034
+__inference_sequential_layer_call_fn_346863ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2г
F__inference_sequential_layer_call_and_return_conditional_losses_347317
F__inference_sequential_layer_call_and_return_conditional_losses_347664
F__inference_sequential_layer_call_and_return_conditional_losses_346904
F__inference_sequential_layer_call_and_return_conditional_losses_346945ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
џBЎ
!__inference__wrapped_model_345163time_distributed_input"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Т2П
1__inference_time_distributed_layer_call_fn_347673
1__inference_time_distributed_layer_call_fn_347682
1__inference_time_distributed_layer_call_fn_347691
1__inference_time_distributed_layer_call_fn_347700ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ю2ы
L__inference_time_distributed_layer_call_and_return_conditional_losses_347724
L__inference_time_distributed_layer_call_and_return_conditional_losses_347748
L__inference_time_distributed_layer_call_and_return_conditional_losses_347763
L__inference_time_distributed_layer_call_and_return_conditional_losses_347778ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ъ2Ч
3__inference_time_distributed_1_layer_call_fn_347787
3__inference_time_distributed_1_layer_call_fn_347796
3__inference_time_distributed_1_layer_call_fn_347805
3__inference_time_distributed_1_layer_call_fn_347814ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ж2Г
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_347838
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_347862
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_347877
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_347892ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ъ2Ч
3__inference_time_distributed_2_layer_call_fn_347897
3__inference_time_distributed_2_layer_call_fn_347902
3__inference_time_distributed_2_layer_call_fn_347907
3__inference_time_distributed_2_layer_call_fn_347912ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ж2Г
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_347930
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_347948
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_347957
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_347966ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ъ2Ч
3__inference_time_distributed_3_layer_call_fn_347971
3__inference_time_distributed_3_layer_call_fn_347976
3__inference_time_distributed_3_layer_call_fn_347981
3__inference_time_distributed_3_layer_call_fn_347986ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ж2Г
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_348003
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_348020
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_348030
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_348040ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ч2ф
%__inference_lstm_layer_call_fn_348051
%__inference_lstm_layer_call_fn_348062
%__inference_lstm_layer_call_fn_348073
%__inference_lstm_layer_call_fn_348084’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
г2а
@__inference_lstm_layer_call_and_return_conditional_losses_348313
@__inference_lstm_layer_call_and_return_conditional_losses_348606
@__inference_lstm_layer_call_and_return_conditional_losses_348835
@__inference_lstm_layer_call_and_return_conditional_losses_349128’
ћ≤»
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
–2Ќ
&__inference_dense_layer_call_fn_349137Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_dense_layer_call_and_return_conditional_losses_349148Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_1_layer_call_fn_349157Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_1_layer_call_and_return_conditional_losses_349167Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЏB„
$__inference_signature_wrapper_346980time_distributed_input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_conv2d_layer_call_fn_349176Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_conv2d_layer_call_and_return_conditional_losses_349187Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_conv2d_1_layer_call_fn_349196Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv2d_1_layer_call_and_return_conditional_losses_349207Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
И2Е
.__inference_max_pooling2d_layer_call_fn_349212
.__inference_max_pooling2d_layer_call_fn_349217Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Њ2ї
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_349222
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_349227Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_flatten_layer_call_fn_349232Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_flatten_layer_call_and_return_conditional_losses_349238Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ь2Щ
*__inference_lstm_cell_layer_call_fn_349255
*__inference_lstm_cell_layer_call_fn_349272Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
E__inference_lstm_cell_layer_call_and_return_conditional_losses_349347
E__inference_lstm_cell_layer_call_and_return_conditional_losses_349454Њ
µ≤±
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 ≥
!__inference__wrapped_model_345163НABCDEGF./56KҐH
AҐ>
<К9
time_distributed_input€€€€€€€€€
™ "1™.
,
dense_1!К
dense_1€€€€€€€€€і
D__inference_conv2d_1_layer_call_and_return_conditional_losses_349207lCD7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€	 
Ъ М
)__inference_conv2d_1_layer_call_fn_349196_CD7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ " К€€€€€€€€€	 ≤
B__inference_conv2d_layer_call_and_return_conditional_losses_349187lAB7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ К
'__inference_conv2d_layer_call_fn_349176_AB7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ " К€€€€€€€€€£
C__inference_dense_1_layer_call_and_return_conditional_losses_349167\56/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ {
(__inference_dense_1_layer_call_fn_349157O56/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€°
A__inference_dense_layer_call_and_return_conditional_losses_349148\.//Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ y
&__inference_dense_layer_call_fn_349137O.//Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€ ®
C__inference_flatten_layer_call_and_return_conditional_losses_349238a7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ А
(__inference_flatten_layer_call_fn_349232T7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ "К€€€€€€€€€А»
E__inference_lstm_cell_layer_call_and_return_conditional_losses_349347юEGFБҐ~
wҐt
!К
inputs€€€€€€€€€А
KҐH
"К
states/0€€€€€€€€€@
"К
states/1€€€€€€€€€@
p 
™ "sҐp
iҐf
К
0/0€€€€€€€€€@
EЪB
К
0/1/0€€€€€€€€€@
К
0/1/1€€€€€€€€€@
Ъ »
E__inference_lstm_cell_layer_call_and_return_conditional_losses_349454юEGFБҐ~
wҐt
!К
inputs€€€€€€€€€А
KҐH
"К
states/0€€€€€€€€€@
"К
states/1€€€€€€€€€@
p
™ "sҐp
iҐf
К
0/0€€€€€€€€€@
EЪB
К
0/1/0€€€€€€€€€@
К
0/1/1€€€€€€€€€@
Ъ Э
*__inference_lstm_cell_layer_call_fn_349255оEGFБҐ~
wҐt
!К
inputs€€€€€€€€€А
KҐH
"К
states/0€€€€€€€€€@
"К
states/1€€€€€€€€€@
p 
™ "cҐ`
К
0€€€€€€€€€@
AЪ>
К
1/0€€€€€€€€€@
К
1/1€€€€€€€€€@Э
*__inference_lstm_cell_layer_call_fn_349272оEGFБҐ~
wҐt
!К
inputs€€€€€€€€€А
KҐH
"К
states/0€€€€€€€€€@
"К
states/1€€€€€€€€€@
p
™ "cҐ`
К
0€€€€€€€€€@
AЪ>
К
1/0€€€€€€€€€@
К
1/1€€€€€€€€€@¬
@__inference_lstm_layer_call_and_return_conditional_losses_348313~EGFPҐM
FҐC
5Ъ2
0К-
inputs/0€€€€€€€€€€€€€€€€€€А

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ¬
@__inference_lstm_layer_call_and_return_conditional_losses_348606~EGFPҐM
FҐC
5Ъ2
0К-
inputs/0€€€€€€€€€€€€€€€€€€А

 
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ≤
@__inference_lstm_layer_call_and_return_conditional_losses_348835nEGF@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€А

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ≤
@__inference_lstm_layer_call_and_return_conditional_losses_349128nEGF@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€А

 
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Ъ
%__inference_lstm_layer_call_fn_348051qEGFPҐM
FҐC
5Ъ2
0К-
inputs/0€€€€€€€€€€€€€€€€€€А

 
p 

 
™ "К€€€€€€€€€@Ъ
%__inference_lstm_layer_call_fn_348062qEGFPҐM
FҐC
5Ъ2
0К-
inputs/0€€€€€€€€€€€€€€€€€€А

 
p

 
™ "К€€€€€€€€€@К
%__inference_lstm_layer_call_fn_348073aEGF@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€А

 
p 

 
™ "К€€€€€€€€€@К
%__inference_lstm_layer_call_fn_348084aEGF@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€А

 
p

 
™ "К€€€€€€€€€@м
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_349222ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ µ
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_349227h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€	 
™ "-Ґ*
#К 
0€€€€€€€€€ 
Ъ ƒ
.__inference_max_pooling2d_layer_call_fn_349212СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Н
.__inference_max_pooling2d_layer_call_fn_349217[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€	 
™ " К€€€€€€€€€ ‘
F__inference_sequential_layer_call_and_return_conditional_losses_346904ЙABCDEGF./56SҐP
IҐF
<К9
time_distributed_input€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ‘
F__inference_sequential_layer_call_and_return_conditional_losses_346945ЙABCDEGF./56SҐP
IҐF
<К9
time_distributed_input€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ √
F__inference_sequential_layer_call_and_return_conditional_losses_347317yABCDEGF./56CҐ@
9Ґ6
,К)
inputs€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ √
F__inference_sequential_layer_call_and_return_conditional_losses_347664yABCDEGF./56CҐ@
9Ґ6
,К)
inputs€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ђ
+__inference_sequential_layer_call_fn_346301|ABCDEGF./56SҐP
IҐF
<К9
time_distributed_input€€€€€€€€€
p 

 
™ "К€€€€€€€€€Ђ
+__inference_sequential_layer_call_fn_346863|ABCDEGF./56SҐP
IҐF
<К9
time_distributed_input€€€€€€€€€
p

 
™ "К€€€€€€€€€Ы
+__inference_sequential_layer_call_fn_347007lABCDEGF./56CҐ@
9Ґ6
,К)
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€Ы
+__inference_sequential_layer_call_fn_347034lABCDEGF./56CҐ@
9Ґ6
,К)
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€–
$__inference_signature_wrapper_346980ІABCDEGF./56eҐb
Ґ 
[™X
V
time_distributed_input<К9
time_distributed_input€€€€€€€€€"1™.
,
dense_1!К
dense_1€€€€€€€€€б
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_347838ОCDLҐI
BҐ?
5К2
inputs&€€€€€€€€€€€€€€€€€€
p 

 
™ ":Ґ7
0К-
0&€€€€€€€€€€€€€€€€€€	 
Ъ б
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_347862ОCDLҐI
BҐ?
5К2
inputs&€€€€€€€€€€€€€€€€€€
p

 
™ ":Ґ7
0К-
0&€€€€€€€€€€€€€€€€€€	 
Ъ ќ
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_347877|CDCҐ@
9Ґ6
,К)
inputs€€€€€€€€€
p 

 
™ "1Ґ.
'К$
0€€€€€€€€€	 
Ъ ќ
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_347892|CDCҐ@
9Ґ6
,К)
inputs€€€€€€€€€
p

 
™ "1Ґ.
'К$
0€€€€€€€€€	 
Ъ є
3__inference_time_distributed_1_layer_call_fn_347787БCDLҐI
BҐ?
5К2
inputs&€€€€€€€€€€€€€€€€€€
p 

 
™ "-К*&€€€€€€€€€€€€€€€€€€	 є
3__inference_time_distributed_1_layer_call_fn_347796БCDLҐI
BҐ?
5К2
inputs&€€€€€€€€€€€€€€€€€€
p

 
™ "-К*&€€€€€€€€€€€€€€€€€€	 ¶
3__inference_time_distributed_1_layer_call_fn_347805oCDCҐ@
9Ґ6
,К)
inputs€€€€€€€€€
p 

 
™ "$К!€€€€€€€€€	 ¶
3__inference_time_distributed_1_layer_call_fn_347814oCDCҐ@
9Ґ6
,К)
inputs€€€€€€€€€
p

 
™ "$К!€€€€€€€€€	 Ё
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_347930КLҐI
BҐ?
5К2
inputs&€€€€€€€€€€€€€€€€€€	 
p 

 
™ ":Ґ7
0К-
0&€€€€€€€€€€€€€€€€€€ 
Ъ Ё
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_347948КLҐI
BҐ?
5К2
inputs&€€€€€€€€€€€€€€€€€€	 
p

 
™ ":Ґ7
0К-
0&€€€€€€€€€€€€€€€€€€ 
Ъ  
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_347957xCҐ@
9Ґ6
,К)
inputs€€€€€€€€€	 
p 

 
™ "1Ґ.
'К$
0€€€€€€€€€ 
Ъ  
N__inference_time_distributed_2_layer_call_and_return_conditional_losses_347966xCҐ@
9Ґ6
,К)
inputs€€€€€€€€€	 
p

 
™ "1Ґ.
'К$
0€€€€€€€€€ 
Ъ і
3__inference_time_distributed_2_layer_call_fn_347897}LҐI
BҐ?
5К2
inputs&€€€€€€€€€€€€€€€€€€	 
p 

 
™ "-К*&€€€€€€€€€€€€€€€€€€ і
3__inference_time_distributed_2_layer_call_fn_347902}LҐI
BҐ?
5К2
inputs&€€€€€€€€€€€€€€€€€€	 
p

 
™ "-К*&€€€€€€€€€€€€€€€€€€ Ґ
3__inference_time_distributed_2_layer_call_fn_347907kCҐ@
9Ґ6
,К)
inputs€€€€€€€€€	 
p 

 
™ "$К!€€€€€€€€€ Ґ
3__inference_time_distributed_2_layer_call_fn_347912kCҐ@
9Ґ6
,К)
inputs€€€€€€€€€	 
p

 
™ "$К!€€€€€€€€€ ÷
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_348003ГLҐI
BҐ?
5К2
inputs&€€€€€€€€€€€€€€€€€€ 
p 

 
™ "3Ґ0
)К&
0€€€€€€€€€€€€€€€€€€А
Ъ ÷
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_348020ГLҐI
BҐ?
5К2
inputs&€€€€€€€€€€€€€€€€€€ 
p

 
™ "3Ґ0
)К&
0€€€€€€€€€€€€€€€€€€А
Ъ √
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_348030qCҐ@
9Ґ6
,К)
inputs€€€€€€€€€ 
p 

 
™ "*Ґ'
 К
0€€€€€€€€€А
Ъ √
N__inference_time_distributed_3_layer_call_and_return_conditional_losses_348040qCҐ@
9Ґ6
,К)
inputs€€€€€€€€€ 
p

 
™ "*Ґ'
 К
0€€€€€€€€€А
Ъ ≠
3__inference_time_distributed_3_layer_call_fn_347971vLҐI
BҐ?
5К2
inputs&€€€€€€€€€€€€€€€€€€ 
p 

 
™ "&К#€€€€€€€€€€€€€€€€€€А≠
3__inference_time_distributed_3_layer_call_fn_347976vLҐI
BҐ?
5К2
inputs&€€€€€€€€€€€€€€€€€€ 
p

 
™ "&К#€€€€€€€€€€€€€€€€€€АЫ
3__inference_time_distributed_3_layer_call_fn_347981dCҐ@
9Ґ6
,К)
inputs€€€€€€€€€ 
p 

 
™ "К€€€€€€€€€АЫ
3__inference_time_distributed_3_layer_call_fn_347986dCҐ@
9Ґ6
,К)
inputs€€€€€€€€€ 
p

 
™ "К€€€€€€€€€Ая
L__inference_time_distributed_layer_call_and_return_conditional_losses_347724ОABLҐI
BҐ?
5К2
inputs&€€€€€€€€€€€€€€€€€€
p 

 
™ ":Ґ7
0К-
0&€€€€€€€€€€€€€€€€€€
Ъ я
L__inference_time_distributed_layer_call_and_return_conditional_losses_347748ОABLҐI
BҐ?
5К2
inputs&€€€€€€€€€€€€€€€€€€
p

 
™ ":Ґ7
0К-
0&€€€€€€€€€€€€€€€€€€
Ъ ћ
L__inference_time_distributed_layer_call_and_return_conditional_losses_347763|ABCҐ@
9Ґ6
,К)
inputs€€€€€€€€€
p 

 
™ "1Ґ.
'К$
0€€€€€€€€€
Ъ ћ
L__inference_time_distributed_layer_call_and_return_conditional_losses_347778|ABCҐ@
9Ґ6
,К)
inputs€€€€€€€€€
p

 
™ "1Ґ.
'К$
0€€€€€€€€€
Ъ Ј
1__inference_time_distributed_layer_call_fn_347673БABLҐI
BҐ?
5К2
inputs&€€€€€€€€€€€€€€€€€€
p 

 
™ "-К*&€€€€€€€€€€€€€€€€€€Ј
1__inference_time_distributed_layer_call_fn_347682БABLҐI
BҐ?
5К2
inputs&€€€€€€€€€€€€€€€€€€
p

 
™ "-К*&€€€€€€€€€€€€€€€€€€§
1__inference_time_distributed_layer_call_fn_347691oABCҐ@
9Ґ6
,К)
inputs€€€€€€€€€
p 

 
™ "$К!€€€€€€€€€§
1__inference_time_distributed_layer_call_fn_347700oABCҐ@
9Ґ6
,К)
inputs€€€€€€€€€
p

 
™ "$К!€€€€€€€€€