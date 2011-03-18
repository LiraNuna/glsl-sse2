#ifndef __SWIZZLE2_H__
#define __SWIZZLE2_H__

#define xy shuffle2_rw2<_MM_SHUFFLE2(1,0)>()
#define yx shuffle2_rw2<_MM_SHUFFLE2(0,1)>()

#define xx shuffle2_ro2<_MM_SHUFFLE2(0,0)>()
#define yy shuffle2_ro2<_MM_SHUFFLE2(1,1)>()

#define xxxx shuffle4_ro2<_MM_SHUFFLE(0,0,0,0)>()
#define xxxy shuffle4_ro2<_MM_SHUFFLE(1,0,0,0)>()
#define xxyx shuffle4_ro2<_MM_SHUFFLE(0,1,0,0)>()
#define xxyy shuffle4_ro2<_MM_SHUFFLE(1,1,0,0)>()
#define xyxx shuffle4_ro2<_MM_SHUFFLE(0,0,1,0)>()
#define xyxy shuffle4_ro2<_MM_SHUFFLE(1,0,1,0)>()
#define xyyx shuffle4_ro2<_MM_SHUFFLE(0,1,1,0)>()
#define xyyy shuffle4_ro2<_MM_SHUFFLE(1,1,1,0)>()
#define yxxx shuffle4_ro2<_MM_SHUFFLE(0,0,0,1)>()
#define yxxy shuffle4_ro2<_MM_SHUFFLE(1,0,0,1)>()
#define yxyx shuffle4_ro2<_MM_SHUFFLE(0,1,0,1)>()
#define yxyy shuffle4_ro2<_MM_SHUFFLE(1,1,0,1)>()
#define yyxx shuffle4_ro2<_MM_SHUFFLE(0,0,1,1)>()
#define yyxy shuffle4_ro2<_MM_SHUFFLE(1,0,1,1)>()
#define yyyx shuffle4_ro2<_MM_SHUFFLE(0,1,1,1)>()
#define yyyy shuffle4_ro2<_MM_SHUFFLE(1,1,1,1)>()

#endif
