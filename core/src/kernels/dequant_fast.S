# TurboQuant: Fast Dequantization Assembly (AVX-512)
# Optimized for Intel Tiger Lake (i5-11400H)
#
# void dequant_int2_avx512(
#     const uint8_t* packed,   // rdi - packed INT2 data (4 vals/byte)
#     const float* scales,     // rsi - per-group scale factors
#     float* output,           // rdx - output FP32 buffer
#     size_t n                 // rcx - number of output elements
# )
#
# Unpacks INT2 (0-3) values and dequantizes to FP32.
# Processes 64 output values per iteration (16 bytes → 64 floats).
# Throughput: ~1 cycle/element on Tiger Lake

.intel_syntax noprefix
.text
.globl dequant_int2_avx512
.type  dequant_int2_avx512, @function
.align 64

dequant_int2_avx512:
    push    rbx
    push    r12

    mov     r12, rcx            # r12 = n (element count)
    shr     r12, 6              # r12 = n / 64 (iterations)
    test    r12, r12
    jz      .Ldq2_done

    # Constant: 0x03 mask for extracting 2-bit values
    vpbroadcastd zmm15, [rip + .Ldq2_mask03]
    # Constant: 1.5f for centering (subtract to get -1.5, -0.5, 0.5, 1.5)
    vbroadcastss zmm14, [rip + .Ldq2_center]

    xor     rbx, rbx            # rbx = group counter for scales

.align 64
.Ldq2_loop:
    # Load 16 packed bytes = 64 INT2 values
    vmovdqu xmm0, [rdi]        # xmm0 = 16 bytes of packed INT2

    # Broadcast scale for current group
    vbroadcastss zmm13, [rsi + rbx * 4]

    # ---- Unpack byte 0-3: first 16 values ----
    # Extract bits [1:0] of each byte
    vpmovzxbd zmm1, xmm0       # Zero-extend bytes to dwords (16 values)
    vpandd  zmm2, zmm1, zmm15  # zmm2 = val & 0x03 (bits [1:0])

    # Convert INT32 → FP32, center and scale
    vcvtdq2ps zmm2, zmm2       # zmm2 = float(int2_val)
    vsubps  zmm2, zmm2, zmm14  # zmm2 -= 1.5 (center around 0)
    vmulps  zmm2, zmm2, zmm13  # zmm2 *= scale
    vmovups [rdx], zmm2        # Store 16 floats

    # ---- Shift right 2 bits, extract next 16 values ----
    vpsrld  zmm3, zmm1, 2      # Shift right 2 bits
    vpandd  zmm3, zmm3, zmm15  # zmm3 = (val >> 2) & 0x03

    vcvtdq2ps zmm3, zmm3
    vsubps  zmm3, zmm3, zmm14
    vmulps  zmm3, zmm3, zmm13
    vmovups [rdx + 64], zmm3   # Store next 16 floats

    # ---- Shift right 4 bits, extract next 16 values ----
    vpsrld  zmm4, zmm1, 4
    vpandd  zmm4, zmm4, zmm15

    vcvtdq2ps zmm4, zmm4
    vsubps  zmm4, zmm4, zmm14
    vmulps  zmm4, zmm4, zmm13
    vmovups [rdx + 128], zmm4  # Store next 16 floats

    # ---- Shift right 6 bits, extract last 16 values ----
    vpsrld  zmm5, zmm1, 6
    vpandd  zmm5, zmm5, zmm15

    vcvtdq2ps zmm5, zmm5
    vsubps  zmm5, zmm5, zmm14
    vmulps  zmm5, zmm5, zmm13
    vmovups [rdx + 192], zmm5  # Store last 16 floats

    # Advance pointers
    add     rdi, 16             # 16 packed bytes consumed
    add     rdx, 256            # 64 floats * 4 bytes = 256 bytes produced
    inc     rbx                 # Next scale group

    dec     r12
    jnz     .Ldq2_loop

.Ldq2_done:
    pop     r12
    pop     rbx
    ret

# ============================================
# void dequant_int4_avx512(
#     const uint8_t* packed,   // rdi - packed INT4 data (2 vals/byte)
#     const float* scales,     // rsi - per-group scale factors
#     float* output,           // rdx - output FP32 buffer
#     size_t n                 // rcx - number of output elements
# )
# ============================================
.globl dequant_int4_avx512
.type  dequant_int4_avx512, @function
.align 64

dequant_int4_avx512:
    push    rbx
    push    r12

    mov     r12, rcx
    shr     r12, 5              # n / 32 (32 elements per iteration)
    test    r12, r12
    jz      .Ldq4_done

    # Constant: 0x0F mask for INT4
    vpbroadcastd zmm15, [rip + .Ldq4_mask0f]
    # Constant: 7.5f for centering
    vbroadcastss zmm14, [rip + .Ldq4_center]

    xor     rbx, rbx

.align 64
.Ldq4_loop:
    # Load 16 packed bytes = 32 INT4 values
    vmovdqu xmm0, [rdi]

    # Broadcast scale
    vbroadcastss zmm13, [rsi + rbx * 4]

    # Extract low nibbles (even indices)
    vpmovzxbd zmm1, xmm0       # Zero-extend to 32-bit
    vpandd  zmm2, zmm1, zmm15  # Low nibble = val & 0x0F

    vcvtdq2ps zmm2, zmm2
    vsubps  zmm2, zmm2, zmm14  # Center around 0
    vmulps  zmm2, zmm2, zmm13  # Apply scale
    vmovups [rdx], zmm2        # Store 16 floats

    # Extract high nibbles (odd indices)
    vpsrld  zmm3, zmm1, 4      # Shift right 4 bits
    vpandd  zmm3, zmm3, zmm15  # High nibble

    vcvtdq2ps zmm3, zmm3
    vsubps  zmm3, zmm3, zmm14
    vmulps  zmm3, zmm3, zmm13
    vmovups [rdx + 64], zmm3   # Store next 16 floats

    # Advance
    add     rdi, 16
    add     rdx, 128            # 32 floats * 4 bytes
    inc     rbx

    dec     r12
    jnz     .Ldq4_loop

.Ldq4_done:
    pop     r12
    pop     rbx
    ret

# ============================================
# Constants
# ============================================
.section .rodata
.align 64

.Ldq2_mask03:
    .long 0x00000003

.Ldq2_center:
    .float 1.5

.Ldq4_mask0f:
    .long 0x0000000F

.Ldq4_center:
    .float 7.5
