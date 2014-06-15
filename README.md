# glsl-sse2

glsl-sse2 is a header-only abstraction library aimed providing the comfort
of GLSL programming language and efficiency of SSE2. In short, it is an optimized
SIMD vector library that behaves like the GLSL shading language.

## Why?

SSE2 is a very powerful extension used in the ia32 and amd64 instruction set
often overlooked by many programmers who are seeking a performance boost for
their applications.

The programmers who are aware of SSE, either lack the knowledge and know-how to
fully utilize it, or are forced to write unportable, unmaintainable assembly
code, or use SSE intrinsics. SSE intrinsic based code quality depends heavily on
the compiler's quality and very often produces poor quality code that it's
scalar equivalent will out-perform (for more information on the subject,
see http://www.liranuna.com/?p=984).

glsl-sse2 takes care of all the nasty stuff while providing the familiar API of
GLSL - no ugly assembly no cache misses due to bad compiler output and no need
to resort to unmaintainable or unportable code.

## Example code and output

glsl-sse2 is geared toward performance, and will do it's best to signal the
compiler how to best utilize the situation at hand:
This example of a cross product between two `vec4`s (this code assumes the
`w` component is meaningless):

    vec4 cross(const vec4 &a, const vec4 &b)
    {
	    return a.yzxw * b.zxyw - a.zxyw * b.yzxw;
    }

would be converted to this assembly code using glsl-sse2 and GCC 4.4:

    _Z5crossRK4vec4S1_:
        movaps    (%rsi), %xmm1
        movaps    (%rdx), %xmm2
        pshufd    $201, %xmm1, %xmm5
        pshufd    $210, %xmm2, %xmm0
        pshufd    $210, %xmm1, %xmm4
        pshufd    $201, %xmm2, %xmm3
        mulps     %xmm0, %xmm5
        mulps     %xmm3, %xmm4
        subps     %xmm4, %xmm5
        movaps    %xmm5, (%rdi)
        ret

Minimal instructions and compiler hints help GCC to output a hand-written 
quality assembly code, complete with instruction pairing and little overhead.
        
## Getting maximum performance out of glsl-sse2

glsl-sse2 is already written in a way that will try and make the compiler output
the best possible code. However, some compilers are unable to produce the best
possible code - compilers that ignore instruction pairing hints and parallel
operations. It is best to avoid those compilers if you are seeking high
performance.
Most notable compiler that produces poor code is MSVC 2008 and below.

Another factor that will effect code performance is the architecture. glsl-sse2
will be most effective in a 64bit environment where there are double the SSE
registers to operate on, causing less register pressure, as well as a guarentee
of SSE2 being present - without the need to check CPUID (although most, if not
all, CPUs today support SSE2 which was released in 2001, a decade ago!).

## Status

Currently the project is in a stable state, however it had not been subjected
to real-world (ab)use yet. You are welcome to try it at your own risk, and file 
bug reports, if such are found.

## What about `vec3`? Where is `mat2`?

Because of alignment issues, only direct SSE2 compatible types will be
implemented. Types such as `vec3`, `mat3` and types that depend on those,
will not be implemented.

If you need an unsupported type, simply use a bigger one - you will still get 
the best out of the vectorization power of SSE (there are, of course, several
exceptions, such as `mat2` and `vec2`, depending on the operations performed).

## Tested compilers

 * GNU C Compiler 4.x and above
 * Microsoft Visual C++ 2008 and better
 * Intel C++ Compiler 10.0 and above
 * clang 2.8
 * llvm-gcc using DragonEgg 2.8

## Recommended compilers

Since not all compilers are equal, and behave differently, glsl-sse2 does its
best to try and make most of the compilers output similar code. However some
compilers take hints better than others. These compilers are best suited to
extract most performance out of glsl-sse2, ordered from best to worst:

 * Intel C++ Compiler 12.0+
 * GNU C Compiler 4.4+
 * Microsoft Visual C++ 2010+
 * clang 2.8 / DragonEgg
 
The use of LLVM compilers (clang, dragonegg) is highly discouraged, as LLVM does
not output code that makes good use of instruction pairing, which can result a 
flat 100% speed increase when used correctly (such as ICC/GCC).
 
## TODO
 - Downswizzle `dvec4` => `dvec2` (Circular dependency issues) 
 - Refactor swizzling, it's a mess (`new_swizzle` branch only works on GCC 4.4)
 - Conversion functions between vectors
 - Missing classes: `dmat2x4`, `dmat4x2`
 - Better tests
 - Namespacing
 - Complete rewrite of `bvec4`
 	- swizzling of booleans
	- `bvec4` generators for all vector classes
 - Exponential functions for `vec4`, `dvec4`
 - Trigonometric functions for `vec4`, `dvec4`
 - Division for `ivec4`, `uvec4`
