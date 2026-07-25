#!/usr/bin/env python3
"""Encode CacheFlex scalar and SVE assembly mnemonics as ``.inst`` words.

Supported source forms are the explicit scalar form
``SPM{LDR,STR,CP,WB}_<bytes>_{IMM,PRE,POST}`` and the ``spm.ld1*`` /
``spm.st1*`` vector forms emitted by the supplied kernels.  Ordinary assembly
lines pass through unchanged.  Any detected custom instruction that cannot be
encoded terminates the build with a nonzero status.
"""

import os
import re
import sys


def debug_print(*args, **kwargs):
    """Emit encoder diagnostics only when explicitly requested."""
    if os.environ.get("CACHEFLEX_ENCODE_DEBUG"):
        print(*args, **kwargs)

# SVE/SPM vector instruction mappings.  The dtype values match gem5's
# decodeSpmContigLoadSIInsts() and decodeSpmContigStoreSIInsts().
SPM_VECTOR_OPS = {
    # ---- Contiguous ----
    "ld1b":  {"type": "ld", "dtype": 0x0},   # Base<uint8_t ,  uint8_t>  full-width byte load
    "ld1h":  {"type": "ld", "dtype": 0x5},   # Base<uint16_t, uint16_t> full-width half load
    "ld1w":  {"type": "ld", "dtype": 0xa},   # Base<uint32_t, uint32_t> full-width word load
    "ld1d":  {"type": "ld", "dtype": 0xf},   # Base<uint64_t, uint64_t> full-width dword load (=ld1qd)

    "st1b":  {"type": "st", "dtype": 0x0},   # Base<uint8_t,  uint8_t>  full-width byte store
    "st1h":  {"type": "st", "dtype": 0x5},   # Base<uint16_t, uint16_t> full-width half store
    "st1w":  {"type": "st", "dtype": 0xa},   # Base<uint32_t, uint32_t> full-width word store
    "st1d":  {"type": "st", "dtype": 0xf},   # Base<uint64_t, uint64_t> full-width dword store

    # ---- Extended signed/int variants (load only) ----
    "ld1sd": {"type": "ld", "dtype": 0x4},  # Base<int64_t ,  int32_t>
    "ld1sh": {"type": "ld", "dtype": 0x8},  # Base<int64_t ,  int16_t>
    "ld1sw": {"type": "ld", "dtype": 0x9},  # Base<int32_t ,  int16_t>
    "ld1sb": {"type": "ld", "dtype": 0xc},  # Base<int64_t ,  int8_t >
    "ld1sbw":{"type": "ld", "dtype": 0xd},  # Base<int32_t ,  int8_t >
    "ld1sbh":{"type": "ld", "dtype": 0xe},  # Base<int16_t ,  int8_t >


    "ld1hh": {"type": "ld", "dtype": 0x5},   # uint16 → uint16
    "ld1wh": {"type": "ld", "dtype": 0x6},   # uint32 → uint16
    "ld1dh": {"type": "ld", "dtype": 0x7},   # uint64 → uint16

    # ---- Replicate family (ld1rq*) ----
    # dtype = msz_esz: msz in bits[24:23], esz=00 in bits[22:21]
    "ld1rqb": {"type": "ldrq", "dtype": 0x0},  # msz=00, esz=00 → 0b0000
    "ld1rqh": {"type": "ldrq", "dtype": 0x4},  # msz=01, esz=00 → 0b0100
    "ld1rqw": {"type": "ldrq", "dtype": 0x8},  # msz=10, esz=00 → 0b1000
    "ld1rqd": {"type": "ldrq", "dtype": 0xC},  # msz=11, esz=00 → 0b1100

    # ---- Broadcast family (ld1ro*) ----
    "ld1rob": {"type": "ldro", "dtype": 0x1},  # msz=00, esz=01 → 0b0001
    "ld1roh": {"type": "ldro", "dtype": 0x5},  # msz=01, esz=01 → 0b0101
    "ld1row": {"type": "ldro", "dtype": 0x9},  # msz=10, esz=01 → 0b1001
    "ld1rod": {"type": "ldro", "dtype": 0xD},  # msz=11, esz=01 → 0b1101

    # ---- Large-width contiguous variants (for completeness) ----
    # dtype 0xa, 0xb, 0xf all appear in load/store cases
    "ld1aw":  {"type": "ld", "dtype": 0xa},  # Base<uint32_t, uint32_t>
    "ld1ad":  {"type": "ld", "dtype": 0xb},  # Base<uint64_t, uint32_t>
    "ld1qd":  {"type": "ld", "dtype": 0xf},  # Base<uint64_t, uint64_t>

    "st1aw":  {"type": "st", "dtype": 0xa},  # Base<uint32_t, uint32_t>
    "st1ad":  {"type": "st", "dtype": 0xb},  # Base<uint64_t, uint32_t>
    "st1qd":  {"type": "st", "dtype": 0xf},  # Base<uint64_t, uint64_t>
}

# Size mappings for scalar operations
# 1/2/4/8 are for LDR/STR (size2 0-3)
# 16/32/64 are for CP/WB (size2=3 + ext2=0-2)
SIZE_MAP = {
    "1": 0, "2": 1, "4": 2, "8": 3, 
    "16": 3, "32": 3, "64": 3, 
}

def parse_register(reg_str):
    """Parses a register string and returns its number."""
    reg_str = reg_str.lower().strip()
    if reg_str == "sp": return 31
    if reg_str in ("xzr", "wzr"): return 31
    m = re.fullmatch(r'z(\d+)', reg_str)
    if m and 0 <= int(m.group(1)) <= 31: return int(m.group(1))
    m = re.fullmatch(r'p(\d+)', reg_str)
    if m and 0 <= int(m.group(1)) <= 15: return int(m.group(1))
    m = re.fullmatch(r'[wx](\d+)', reg_str)
    if m and 0 <= int(m.group(1)) <= 31: return int(m.group(1))
    raise ValueError(f"Invalid register format: {reg_str}")

def parse_spm_vector_instruction(instruction):
    # Return dtype, not size.
    pattern = r'\s*spm\.(\w+)\s+\{?z(\d+)\.(\w+)\}?,\s*p(\d+)(?:/z)?,\s*\[([\w\d]+)(?:,\s*#?(-?\d+))?(?:,\s*mul\s+vl)?\]\s*'
    m = re.fullmatch(pattern, instruction, re.IGNORECASE)
    if m:
        opcode, zt, elem_type, pg, rn, imm = m.groups()
        imm = int(imm) if imm else 0
        opcode_l = opcode.lower()
        if opcode_l not in SPM_VECTOR_OPS:
            raise ValueError(f"Unknown vector opcode: {opcode}")
        zt_num = int(zt)
        pg_num = int(pg)
        if not 0 <= zt_num <= 31:
            raise ValueError(f"Invalid vector register: z{zt}")
        # The encoding has only three predicate-register bits.  Reject p8-p15
        # instead of silently aliasing them to p0-p7.
        if not 0 <= pg_num <= 7:
            raise ValueError(f"Predicate register p{pg} is not encodable (expected p0-p7)")
        vec_info = SPM_VECTOR_OPS[opcode_l]
        return {
            'format': 'vector',
            'opcode': opcode_l,
            'type': vec_info['type'],      # 'ld', 'st', 'ldrq', 'ldro'
            'dtype': vec_info['dtype'],
            'zt': zt_num,
            'pg': pg_num,
            'rn': parse_register(rn),
            'imm': imm,
            'elem_type': elem_type
        }
    return None

def encode_spm_vector_instruction(c):
    """
    Matches the C++ decodeSpmContigLoadSIInsts() / decodeSpmContigStoreSIInsts().

    bits summary:
      bit31 = 1
      bit26 = 1
      bits30:29 = 01 (load) / 11 (store)
      bits24:21 = dtype (0x0..0xF)
      bit20 = 1 for contiguous (ld/st), 0 for broadcast (ldrq/ldro)
      bits19:16 = imm4 (signed, scaled)
      bits15:13 = 0b100 (SPM)
      [12:10]=pg, [9:5]=rn, [4:0]=zt
    """

    zt   = c['zt'] & 0x1F   # Zt
    pg   = c['pg'] & 0x7    # Pg
    rn   = c['rn'] & 0x1F   # Xn
    imm  = int(c.get('imm', 0))
    dtype = c['dtype'] & 0xF  # use dtype directly (0x0..0xF)
    op_type = c['type']       # 'ld', 'st', 'ldrq', 'ldro'

    code = 0

    # --- top-level dispatch shape ---
    code |= (1 << 31)   # bit31 = 1 (enter SVE decoder)
    code |= (1 << 26)   # bit26 = 1 (SPM path)
    # bits27,28,25 = 0 by default

    # --- load/store kind bits[30:29] ---
    if op_type.startswith("ld"):
        code |= (0b01 << 29)
    elif op_type.startswith("st"):
        code |= (0b11 << 29)
    else:
        raise ValueError(f"Unsupported vector type '{op_type}'")

    # --- sub-decode selector bits[15:13] = 0b100 ---
    code |= (0b100 << 13)

    # --- contiguous vs replicate/broadcast ---
    # bit20 = 1 for contiguous (ld1d/st1d)
    # bit20 = 0 for replicate/broadcast (ldrq, ldro)
    if op_type in ("ldrq", "ldro"):
        pass
    else:
        code |= (1 << 20)

    # --- dtype encoding (bits[24:21]) ---
    code |= ((dtype & 0xF) << 21)

    # --- imm4 signed offset, scaled by element size (low 2 bits of dtype) ---
    elem_size = 1 << (dtype & 0x3)  # 0=b(1B),1=h(2B),2=w(4B),3=d(8B)
    if imm % elem_size:
        raise ValueError(
            f"Immediate must be aligned to element size: imm={imm}, "
            f"elem_size={elem_size}"
        )
    imm_elem = imm // elem_size
    if not -8 <= imm_elem <= 7:
        raise ValueError(f"Immediate out of range: imm={imm}, elem_size={elem_size}, imm_elem={imm_elem}")
    code |= ((imm_elem & 0xF) << 16)

    # --- registers ---
    code |= ((pg & 0x7) << 10)
    code |= ((rn & 0x1F) << 5)
    code |= (zt & 0x1F)

    # Optional diagnostic dump.
    def g(lo, hi=None):
        if hi is None:
            hi = lo
        mask = (1 << (hi - lo + 1)) - 1
        return (code >> lo) & mask

    if os.environ.get("CACHEFLEX_ENCODE_DEBUG"):
        print("[FINAL ENCODE]")
        print(f" code = 0x{code:08X}")
        print(f" bit31       = {g(31)}")
        print(f" bits30:29   = {g(29,30):02b} ({'ld' if op_type.startswith('ld') else 'st'})")
        print(f" bit28       = {g(28)}")
        print(f" bit27       = {g(27)}")
        print(f" bit26       = {g(26)}")
        print(f" bits24:21   = 0x{g(21,24):X} (dtype={dtype})")
        print(f" bit20       = {g(20)} (1=contig,0=rep/bcast)")
        print(f" bits19:16   = {g(16,19):04b} (imm_elem={imm_elem})")
        print(f" bits15:13   = {g(13,15):03b} (should be 100)")
        print(f" pg={g(10,12)} rn={g(5,9)} zt={g(0,4)}")
        print("--------------")

    return code
def parse_spm_instruction(instruction):
    """Parses any SPM instruction (scalar or vector) and returns its components."""
    instruction = instruction.strip()

    # Scalar mnemonics are checked before vector mnemonics.
    pattern_full = r'(SPM(CP|WB|LDR|STR))_(\d+)_(\w+)\s+([\w\d]+),\s*\[([\w\d]+)(?:,\s*#?(-?\d+))?\]\s*'
    m_full = re.fullmatch(pattern_full, instruction, re.IGNORECASE)
    if m_full:
        mnemonic, instr_type, size_str, mode, rt, rn, imm = \
            m_full.groups()[0], m_full.groups()[1].lower(), m_full.groups()[2], \
            m_full.groups()[3].lower(), m_full.groups()[4], m_full.groups()[5], m_full.groups()[6]
        if mode not in ("imm", "post", "pre"):
            raise ValueError(f"Unsupported address mode: {mode}")
        return {
            'format': 'scalar',
            'type': instr_type,
            'size': size_str,
            'mode': mode.lower(),
            'rt': parse_register(rt),
            'rn': parse_register(rn),
            'imm': int(imm) if imm else 0
        }

    vec_result = parse_spm_vector_instruction(instruction)
    if vec_result:
        return vec_result

    raise ValueError(f"Unrecognized instruction format: {instruction}")

# ext2-to-block-size mapping; this must match gem5's decoder.
EXT2_LARGE_MAP = {
    "8":  0,   # ext2 = 0 →  8B
    "16": 1,   # ext2 = 1 → 16B
    "32": 2,   # ext2 = 2 → 32B
    "64": 3,   # ext2 = 3 → 64B
}

def encode_spm_scalar_instruction(c):
    """
    Encode SPM instruction (LDR/STR/CP/WB).
    Matches the C++ decodeSPM semantics:
      - LDR/STR use elemBytes = 1,2,4,8
      - CP/WB use elemBytes = 8,16,32,64  from ext2
    """

    debug_print("\n==================== SPM ENCODE DEBUG ====================")
    debug_print("RAW INPUT c =", c)

    op_type = c['type']   # 'ldr', 'str', 'cp', 'wb'

    # ============================================================
    #  size2 FIELD (1,2,4,8)
    # ============================================================
    raw_size = str(c.get('size'))
    debug_print(f"[DEBUG] raw_size = {raw_size}")

    if raw_size not in SIZE_MAP:
        raise ValueError(f"Invalid size {raw_size}")
    if op_type in ("ldr", "str") and raw_size not in ("1", "2", "4", "8"):
        raise ValueError(f"{op_type.upper()} does not support size {raw_size}")
    if op_type in ("cp", "wb") and raw_size not in EXT2_LARGE_MAP:
        raise ValueError(f"{op_type.upper()} does not support size {raw_size}")

    size = SIZE_MAP[raw_size]
    debug_print(f"[DEBUG] size2 (bits21:20) = {size:02b}")

    # ============================================================
    #  addrMd FIELD
    # ============================================================
    addrMd_raw = c.get('addr', c.get('mode', 0))
    if isinstance(addrMd_raw, str):
        try:
            addrMd = {"imm": 0, "post": 1, "pre": 2}[addrMd_raw.lower()]
        except KeyError as error:
            raise ValueError(f"Unsupported address mode: {addrMd_raw}") from error
    else:
        addrMd = int(addrMd_raw) & 0x3

    debug_print(f"[DEBUG] addrMd = {addrMd}   (IMM=0, POST=1, PRE=2)")

    # ============================================================
    #  ext2 FIELD (block size for CP/WB)
    # ============================================================
    ext = EXT2_LARGE_MAP.get(raw_size, 0)
    debug_print(f"[DEBUG] ext2 (bits17:16) = {ext}   (0=8B,1=16B,2=32B,3=64B)")

    imm = int(c.get('imm', 0))
    rt  = int(c['rt']) & 0x1F
    rn  = int(c['rn']) & 0x1F

    # ============================================================
    #  op FIELD
    # ============================================================
    if op_type == 'ldr':
        op = 0b00
    elif op_type == 'str':
        op = 0b01
    elif op_type == 'cp':
        op = 0b10
    elif op_type == 'wb':
        op = 0b11
    else:
        raise ValueError(f"Invalid type {op_type}")

    debug_print(f"[DEBUG] op = {op}   (00=LDR,01=STR,10=CP,11=WB)")

    # ============================================================
    #  initialize machine code
    # ============================================================
    code = 0
    code |= (0xFF << 24)
    code |= (op & 0x3)   << 22
    code |= (size & 0x3) << 20
    code |= (addrMd & 0x3) << 18
    code |= (ext & 0x3)  << 16

    # ============================================================
    #  IMM SCALING
    # ============================================================
    if op <= 1:
        # LDR/STR
        elem_bytes = (1 << size)      # 1,2,4,8
    else:
        # CP/WB
        elem_bytes = (8 << ext)       # 8, 16, 32, or 64 bytes

    debug_print(f"[DEBUG] elem_bytes = {elem_bytes}")

    if imm % elem_bytes:
        raise ValueError(
            f"Immediate must be aligned to transfer size: imm={imm}, "
            f"elem_bytes={elem_bytes}"
        )
    imm_units = imm // elem_bytes
    debug_print(f"[DEBUG] imm = {imm} → imm_units = {imm_units}")

    if not (0 <= imm_units <= 63):
        raise ValueError(f"imm out of range: imm={imm}, elem_bytes={elem_bytes}, imm_units={imm_units}")

    code |= (imm_units & 0x3F) << 10
    code |= (rn & 0x1F) << 5
    code |= (rt & 0x1F)

    # ============================================================
    #  CONFLICT CHECK: reject encodings that collide with gem5 m5 pseudo-instructions.
    #  All m5 ops have bits[15:0] == 0x0110 (imm6=0, rn=x8, rt=x16).
    #  gem5 decodeSPM routes any instruction with (machInst & 0xFFFF)==0x0110
    #  to Gem5Op64, so such an SPM instruction would silently be ignored.
    # ============================================================
    if (code & 0xFFFF) == 0x0110:
        raise ValueError(
            f"Encoded SPM instruction 0x{code:08X} has bits[15:0]==0x0110, "
            f"which conflicts with gem5 m5 pseudo-instructions (e.g. m5_work_begin). "
            f"Avoid using rn=x8, rt=x16 with imm6=0 in scalar SPM instructions."
        )

    # ============================================================
    #  BIT DUMP
    # ============================================================
    def g(lo, hi=None):
        if hi is None: hi = lo
        mask = (1 << (hi - lo + 1)) - 1
        return (code >> lo) & mask

    debug_print("\n[ENCODED RESULT]")
    debug_print(f" code=0x{code:08X}")
    debug_print(f" bits31:24 = {g(24,31):08b}")
    debug_print(f" op(bits23:22)={g(22,23):02b}")
    debug_print(f" size(bits21:20)={g(20,21):02b}")
    debug_print(f" addrMd(bits19:18)={g(18,19):02b}")
    debug_print(f" ext(bits17:16)={g(16,17):02b}")
    debug_print(f" imm6(bits15:10)={g(10,15):06b}")
    debug_print(f" rn(bits9:5)={g(5,9):05b}")
    debug_print(f" rt(bits4:0)={g(0,4):05b}")
    debug_print("===========================================================\n")

    return code

def compile_instruction(instruction):
    """Compiles a single instruction string into its integer machine code."""
    try:
        components = parse_spm_instruction(instruction)
        if components['format'] == 'vector':
            return encode_spm_vector_instruction(components)
        else:
            return encode_spm_scalar_instruction(components)
    except Exception as e:
        # Re-raise with line info is better handled in process_file
        raise ValueError(f"Failed to compile '{instruction}': {e}")

def process_file(filename):
    """Reads an assembly file, replaces SPM instructions, and returns the new lines."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    output_lines = []
    
    # Include known legacy spellings in detection so they fail closed instead
    # of being passed through to the assembler.
    spm_patterns = [
        re.compile(r'^\s*SPM(CP|WB|LDR|STR)_\d+_\w+\b', re.IGNORECASE),
        re.compile(r'^\s*spm\.(ldr|str|cp|wb)\b', re.IGNORECASE),
        re.compile(r'^\s*(spm_ldr|spm_str|cp2spm|spm_wb)\b', re.IGNORECASE),
        re.compile(r'^\s*spm\.(ld1|st1|ld1r)', re.IGNORECASE),
    ]
    
    for line_num, line in enumerate(lines, 1):
        clean_line = line.strip()
        is_spm = False
        
        # Check if a line is a potential SPM instruction, ignoring comments and directives
        if clean_line and not clean_line.startswith(('//', '#', '.')):
            is_spm = any(pattern.match(clean_line) for pattern in spm_patterns)
        
        if is_spm:
            debug_print(f"[DEBUG] Line {line_num}: Detected SPM -> {clean_line}")

            # Handle comments
            parts = clean_line.split('//', 1)
            instruction = parts[0].strip()
            comment_part = " // " + parts[1].strip() if len(parts) > 1 else ""
            
            try:
                if instruction:
                    code = compile_instruction(instruction)
                    
                    # Preserve original indentation
                    indentation = line[:len(line) - len(line.lstrip())]
                    new_line = f"{indentation}.inst 0x{code:08X}{comment_part}\n"
                    output_lines.append(new_line)
                else:
                    output_lines.append(line)
            except Exception as e:
                # An unencoded custom instruction must never be allowed into
                # the assembler output.  Fail closed so a build cannot report
                # success after silently preserving a malformed instruction.
                raise ValueError(
                    f"{filename}:{line_num}: failed to encode "
                    f"'{instruction}': {e}"
                ) from e
        else:
            # Not an SPM instruction, so keep the line as is
            output_lines.append(line)
    
    return output_lines

def main():
    if len(sys.argv) < 2:
        print("Usage: python spm_compiler.py <inputfile.s> [outputfile.s]")
        print("If no output file is specified, writes to standard output.")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        output_lines = process_file(input_file)
        
        if len(sys.argv) >= 3:
            output_file = sys.argv[2]
            with open(output_file, 'w') as f:
                f.writelines(output_lines)
            print(f"Successfully compiled {input_file} to {output_file}")
        else:
            sys.stdout.writelines(output_lines)
            
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during compilation: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
