.globl _start
.globl print_bool
.globl print_int
.globl read_int

_start:
    andq $~15, %rsp
    call main

    # exit(0)
    movl $0x3c, %eax
    xorl %edi, %edi
    syscall
    ud2

error:
    movzxb (%rdi), %rdx
    incq %rdi
    movq %rdi, %rsi
    movl $1, %eax
    movl $2, %edi
    syscall

    movl $0x3c, %eax
    movl $1, %edi
    syscall
    ud2

read_int:
    leaq line_buffer(%rip), %rdi
    movzxb buffer_len(%rip), %rdx
    xorl %ecx, %ecx

    .Lread_loop:
    cmpb %dl, %cl
    jb .Lcheck_char

    # need to read more into the buffer
    pushq %rdi
    pushq %rcx
    xorl %eax, %eax
    movq %rdi, %rsi
    addq %rcx, %rsi
    xorl %edi, %edi
    movl $24, %edx
    subq %rcx, %rdx
    syscall

    testq %rax, %rax
    leaq eof_on_read(%rip), %rdi
    jz error
    leaq fail_to_read(%rip), %rdi
    js error
    popq %rcx
    popq %rdi

    addq %rcx, %rax
    movq %rax, %rdx

    .Lcheck_char:
    movzxb (%rdi,%rcx), %rax
    cmp $'\n', %al
    je .Lfound_newline

    incb %cl
    cmpb $24, %cl
    jne .Lread_loop

    leaq line_too_long(%rip), %rdi
    jmp error

    .Lfound_newline:
    xorl %r10d, %r10d
    mov %dl, buffer_len(%rip)
    movq %rcx, %r9
    xorl %eax, %eax
    leaq (%rdi,%rcx), %rsi
    negq %rcx

    .Lparse_loop:
    movzxb (%rsi,%rcx), %rdx

    cmpb $'-', %dl
    je .Lminus

    subq $'0', %rdx

    cmpb $9, %dl
    leaq not_an_integer(%rip), %r8
    cmovaq %r8, %rdi
    ja error

    shlq $1, %rax
    leaq (%rax,%rax,4), %rax
    addq %rdx, %rax

    incb %cl
    jnz .Lparse_loop

    movq %rax, %rsi
    negq %rsi
    testq %r10, %r10
    cmovnzq %rsi, %rax

    movzxb buffer_len(%rip), %rdx
    subq %r9, %rdx
    decq %rdx
    leaq 1(%rdi,%r9), %rsi
    xorl %ecx, %ecx
    movb %dl, buffer_len(%rip)

    .Lcopy_loop:
    movzxb (%rsi,%rcx), %r8
    movb %r8b, (%rdi,%rcx)

    incb %cl
    cmpb %cl, %dl
    jne .Lcopy_loop
    ret

    .Lminus:
    addq %rcx, %rsi
    cmpq %rsi, %rdi
    leaq not_an_integer(%rip), %r8
    cmovneq %r8, %rdi
    jne error
    subq %rcx, %rsi

    cmpq $2, %r9
    cmovbq %r8, %rdi
    jb error

    incb %r10b

    incb %cl
    jmp .Lparse_loop

print_int:
    subq $8, %rsp
    movq %rsp, %rsi

    decq %rsi
    movb $'\n', %dl
    movb %dl, (%rsi)

    testq %rdi, %rdi
    jnz .Lnot_zero
    decq %rsi
    movb $'0', %dl
    movb %dl, (%rsi)
    jmp .Lprint_int_end

    .Lnot_zero:
    movq %rdi, %rax
    negq %rax
    cmovsq %rdi, %rax
    movl $10, %ecx

    .Lloop:
    xorl %edx, %edx
    decq %rsi
    divq %rcx
    addb $'0', %dl
    movb %dl, (%rsi)

    testq %rax, %rax
    jnz .Lloop

    .Lprint_int_end:
    testq %rdi, %rdi
    jns .Lprint
    decq %rsi
    movb $'-', %dl
    movb %dl, (%rsi)

    .Lprint:
    movl $1, %eax
    movl %eax, %edi
    movq %rsp, %rdx
    subq %rsi, %rdx
    syscall

    addq $8, %rsp
    ret

print_bool:
    leaq falsetrue(%rip), %rsi
    movl $6, %edx
    subl %edi, %edx
    shll $3, %edi
    addq %rdi, %rsi

    movl $1, %eax
    movl %eax, %edi
    syscall

    ret

.section .rodata
falsetrue: .ascii "false\n\x00\x00true\n"

eof_on_read:
.byte eof_on_read_end - eof_on_read - 1
.ascii "got eof when reading int\n"
eof_on_read_end:

fail_to_read:
.byte fail_to_read_end - fail_to_read - 1
.ascii "failed to read from stdin\n"
fail_to_read_end:

line_too_long:
.byte line_too_long_end - line_too_long - 1
.ascii "tried to read 64-bit integer but line was too long\n"
line_too_long_end:

not_an_integer:
.byte not_an_integer_end - not_an_integer - 1
.ascii "tried to read an integer. did not get an integer\n"
not_an_integer_end:

.lcomm line_buffer, 24
.lcomm buffer_len, 1
