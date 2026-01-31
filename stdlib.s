.globl _start
.globl print_bool
.globl print_int

_start:
    andq $~15, %rsp
    call main

    # exit(0)
    movl $0x3c, %eax
    xorl %edi, %edi
    syscall

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
    jmp .Lend

    .Lnot_zero:
    movq %rdi, %rax
    negq %rax
    cmovs %rdi, %rax
    movl $10, %ecx

    .Lloop:
    xorl %edx, %edx
    decq %rsi
    divq %rcx
    addb $'0', %dl
    movb %dl, (%rsi)

    testq %rax, %rax
    jnz .Lloop

    .Lend:
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

.data
falsetrue: .ascii "false\n\x00\x00true\n"
