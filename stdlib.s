.globl _start
.globl print_bool

_start:
    andq $~15, %rsp
    call main

    # exit(0)
    movl $0x3c, %eax
    xorl %edi, %edi
    syscall

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
