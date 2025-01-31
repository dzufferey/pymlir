""" Implementation of the Standard dialect. """

import inspect
import sys
from mlir.dialect import (Dialect, DialectOp, UnaryOperation, BinaryOperation,
                          is_op)
import mlir.astnodes as mast
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


Literal = Union[mast.StringLiteral, float, int, bool]
SsaUse = Union[mast.SsaId, Literal]


# Terminator Operations
@dataclass
class BrOperation(DialectOp):
    block: mast.BlockId
    args: Optional[List[Tuple[mast.SsaId, mast.Type]]] = None
    _syntax_ = ['br {block.BLOCK_ID}',
                'br {block.BLOCK_ID} {args.block_arg_list}']


@dataclass
class CondBrOperation(DialectOp):
    cond: SsaUse
    block_true: mast.BlockId
    block_false: mast.BlockId
    _syntax_ = ['cond_br {cond.ssa_use} , {block_true.BLOCK_ID} , {block_false.BLOCK_ID}']


# Core Operations
@dataclass
class DimOperation(DialectOp):
    operand: mast.SsaId
    index: mast.SsaId
    type: mast.Type
    _syntax_ = 'dim {operand.SSA_ID} , {index.SSA_ID} : {type.type}'


# Memory Operations
@dataclass
class AllocOperation(DialectOp):
    args: mast.DimAndSymbolList
    type: mast.MemRefType
    _syntax_ = 'alloc {args.dim_and_symbol_use_list} : {type.memref_type}'


@dataclass
class AllocStaticOperation(DialectOp):
    base: int
    type: mast.MemRefType
    _syntax_ = 'alloc_static ( {base.INT} ) : {type.memref_type}'


@dataclass
class DeallocOperation(DialectOp):
    arg: SsaUse
    type: mast.MemRefType
    _syntax_ = 'dealloc {arg.ssa_use} : {type.memref_type}'


@dataclass
class DmaStartOperation(DialectOp):
    src: SsaUse
    src_index: List[SsaUse]
    dst: SsaUse
    dst_index: List[SsaUse]
    size: SsaUse
    tag: SsaUse
    tag_index: List[SsaUse]
    src_type: mast.MemRefType
    dst_type: mast.MemRefType
    tag_type: mast.MemRefType
    stride: Optional[SsaUse] = None
    transfer_per_stride: Optional[SsaUse] = None
    _syntax_ = [
        'dma_start {src.ssa_use} [ {src_index.ssa_use_list} ] , {dst.ssa_use} [ {dst_index.ssa_use_list} ] , {size.ssa_use} , {tag.ssa_use} [ {tag_index.ssa_use_list} ] : {src_type.memref_type} , {dst_type.memref_type} , {tag_type.memref_type}',
        'dma_start {src.ssa_use} [ {src_index.ssa_use_list} ] , {dst.ssa_use} [ {dst_index.ssa_use_list} ] , {size.ssa_use} , {tag.ssa_use} [ {tag_index.ssa_use_list} ] , {stride.ssa_use} , {transfer_per_stride.ssa_use} : {src_type.memref_type} , {dst_type.memref_type} , {tag_type.memref_type}'
    ]
@dataclass
class DmaWaitOperation(DialectOp):
    tag: SsaUse
    tag_index: List[SsaUse]
    size: SsaUse
    type: mast.MemRefType
    _syntax_ = 'dma_wait {tag.ssa_use} [ {tag_index.ssa_use_list} ] , {size.ssa_use} : {type.memref_type}'


@dataclass
class ExtractElementOperation(DialectOp):
    arg: SsaUse
    index: List[SsaUse]
    type: mast.Type
    _syntax_ = 'extract_element {arg.ssa_use} [ {index.ssa_use_list} ] : {type.type}'


@dataclass
class LoadOperation(DialectOp):
    arg: SsaUse
    index: List[SsaUse]
    type: mast.MemRefType
    _syntax_ = 'load {arg.ssa_use} [ {index.ssa_use_list} ] : {type.memref_type}'


@dataclass
class SplatOperation(DialectOp):
    arg: SsaUse
    type: Union[mast.VectorType, mast.TensorType]
    _syntax_ = 'splat {arg.ssa_use} : {type.type}'  # (vector_type | tensor_type)


@dataclass
class StoreOperation(DialectOp):
    addr: SsaUse
    ref: SsaUse
    index: List[SsaUse]
    type: mast.MemRefType
    _syntax_ = 'store {addr.ssa_use} , {ref.ssa_use} [ {index.ssa_use_list} ] : {type.memref_type}'


@dataclass
class TensorLoadOperation(DialectOp):
    arg: SsaUse
    type: mast.Type
    _syntax_ = 'tensor_load {arg.ssa_use} : {type.type}'


@dataclass
class TensorStoreOperation(DialectOp):
    arg: SsaUse
    type: mast.Type
    _syntax_ = 'tensor_store {src.ssa_use} , {dst.ssa_use} : {type.memref_type}'

# Unary Operations
class AbsfOperation(UnaryOperation): _opname_ = 'math.absf'
class CeilfOperation(UnaryOperation): _opname_ = 'math.ceil'
class CosOperation(UnaryOperation): _opname_ = 'math.cos'
class ExpOperation(UnaryOperation): _opname_ = 'math.exp'
class NegfOperation(UnaryOperation): _opname_ = 'arith.negf'
class TanhOperation(UnaryOperation): _opname_ = 'math.tanh'
class CopysignOperation(UnaryOperation): _opname_ = 'math.copysign'
class SIToFPOperation(UnaryOperation): _opname_ = 'arith.sitofp'

# Arithmetic Operations
class AddiOperation(BinaryOperation): _opname_ = 'arith.addi'
class AddfOperation(BinaryOperation): _opname_ = 'arith.addf'
class AndOperation(BinaryOperation): _opname_ = 'arith.andi'
class DivisOperation(BinaryOperation): _opname_ = 'arith.divsi'
class DiviuOperation(BinaryOperation): _opname_ = 'arith.divui'
class RemisOperation(BinaryOperation): _opname_ = 'arith.remsi'
class RemiuOperation(BinaryOperation): _opname_ = 'arith.remui'
class DivfOperation(BinaryOperation): _opname_ = 'arith.divf'
class MulfOperation(BinaryOperation): _opname_ = 'arith.mulf'
class MulIOperation(BinaryOperation): _opname_ = 'arith.muli'
class SubiOperation(BinaryOperation): _opname_ = 'arith.subi'
class SubfOperation(BinaryOperation): _opname_ = 'arith.subf'
class OrOperation(BinaryOperation): _opname_ = 'arith.ori'
class XorOperation(BinaryOperation): _opname_ = 'arith.xori'


@dataclass
class CmpiOperation(DialectOp):
    comptype: str
    operand_a: mast.SsaId
    operand_b: mast.SsaId
    type: mast.Type
    _syntax_ = 'cmpi {comptype.STRING} , {operand_a.SSA_ID} , {operand_b.SSA_ID} : {type.type}'


@dataclass
class CmpfOperation(DialectOp):
    comptype: str
    operand_a: mast.SsaId
    operand_b: mast.SsaId
    type: mast.Type
    _syntax_ = 'cmpf {comptype.STRING} , {operand_a.SSA_ID} , {operand_b.SSA_ID} : {type.type}'


@dataclass
class ConstantOperation(DialectOp):
    value: Literal
    type: mast.Type
    _syntax_ = 'constant {value.constant_literal} : {type.type}'


@dataclass
class IndexCastOperation(DialectOp):
    arg: SsaUse
    src_type: mast.Type
    dst_type: mast.Type
    _syntax_ = 'index_cast {arg.ssa_use} : {src_type.type} to {dst_type.type}'


@dataclass
class MemrefCastOperation(DialectOp):
    arg: SsaUse
    src_type: mast.Type
    dst_type: mast.Type
    _syntax_ = 'memref_cast {arg.ssa_use} : {src_type.type} to {dst_type.type}'


@dataclass
class TensorCastOperation(DialectOp):
    arg: SsaUse
    src_type: mast.Type
    dst_type: mast.Type
    _syntax_ = 'tensor_cast {arg.ssa_use} : {src_type.type} to {dst_type.type}'


@dataclass
class SelectOperation(DialectOp):
    cond: SsaUse
    arg_true: SsaUse
    arg_false: SsaUse
    _syntax_ = 'select {cond.ssa_use} , {arg_true.ssa_use} , {arg_false.ssa_use} : {type.type}'


@dataclass
class SubviewOperation(DialectOp):
    operand: SsaUse
    offsets: List[SsaUse]
    sizes: List[SsaUse]
    strides: List[SsaUse]
    src_type: mast.Type
    dst_type: mast.Type
    _syntax_ = 'subview {operand.ssa_use} [ {offsets.ssa_use_list} ] [ {sizes.ssa_use_list} ] [ {strides.ssa_use_list} ] : {src_type.type} to {dst_type.type}'


@dataclass
class ViewOperation(DialectOp):
    operand: SsaUse
    offset: SsaUse
    src_type: mast.Type
    dst_type: mast.Type
    sizes: Optional[List[SsaUse]] = None
    _syntax_ = ['view {operand.ssa_use} [ {offset.ssa_use} ] [ {sizes.ssa_use_list} ] : {src_type.type} to {dst_type.type}',
                'view {operand.ssa_use} [ {offset.ssa_use} ] [  ] : {src_type.type} to {dst_type.type}']


# Inspect current module to get all classes defined above
standard = Dialect('standard', ops=[m[1] for m in inspect.getmembers(
    sys.modules[__name__], lambda obj: is_op(obj, __name__))])
