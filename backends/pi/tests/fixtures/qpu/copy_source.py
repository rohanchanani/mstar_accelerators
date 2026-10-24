# Copyright (c) 2019-2020 Idein Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Stage 2 derivative: preserve the loaded 32-bit word exactly.
# Uniforms: positive vector count, source GPU address, destination GPU address.
# This is a test program, not a model kernel or backend special case.
from videocore7.assembler import *
from videocore7.assembler import Assembly, qpu

@qpu
def qpu_copy_words(asm: Assembly) -> None:
    # rf10: Number of vectors to read.
    # rf11: Pointer to the read vectors + eidx * 4.
    # rf12: Pointer to the write vectors + eidx * 4
    eidx(rf13, sig=ldunifrf(rf10))
    nop(sig=ldunifrf(rf11))
    shl(rf13, rf13, 2)
    add(rf11, rf11, rf13, sig=ldunifrf(rf12))
    add(rf12, rf12, rf13)
    mov(rf13, 4)
    shl(rf13, rf13, 4)

    with loop as l:  # noqa: E741
        mov(tmuc, -1)  # reset TMU config
        mov(tmua, rf11, sig=thrsw)
        nop()
        sub(rf10, rf10, 1, cond="pushz")
        nop(sig=ldtmu(rf0))
        l.b(cond="anyna")
        mov(tmud, rf0).add(rf11, rf11, rf13)  # rf11 += 64
        mov(tmua, rf12).add(rf12, rf12, rf13)  # rf12 += 64
        tmuwt()

    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()
