#include "vec4.h"
#include "mat4.h"

#include <stdio.h>
#include <stdint.h>
#include <limits>
#include <assert.h>

void printv(const vec4 &v)
{
	printf("%+.2f, %+.2f, %+.2f, %+.2f\n", v.r, v.g, v.b, v.a);
}

/*
__m128 color32_to_vec4(uint32_t c)
{
	__m128 m = _mm_cvtpu8_ps(_mm_cvtsi32_si64(c));
	m = _mm_shuffle_ps(m, m, 0x1B);
	return _mm_mul_ps(m, _mm_set1_ps(1.0f / 255.0f));
}

uint32_t vec4_to_color32(__m128 m)
{
	m = _mm_shuffle_ps(m, m, 0x1B);
	m = _mm_mul_ps(m, _mm_set1_ps(255.0f));

	__m64 ll = _mm_cvtps_pi16(m);
	ll = _mm_or_si64(ll, _m_psrlqi(ll, 8));
	ll = _mm_and_si64(ll, _m_from_int64(0xFFFF0000FFFFLL));
	ll = _mm_or_si64(ll, _m_psrlqi(ll, 16));
	return _m_to_int(ll);
}
*/

int main()
{
//	vec4 f1 = vec4(-128.1024f, -0.f, 0, 512.256);

//	printv(f1);
//	printv(sign(f1));
/*
	printv(f1);
	printv(floor(f1));
	printv(ceil(f1));
	printf("\n");
	printf("floorf %+.2f, %+.2f, %+.2f, %+.2f\n", floorf(f1.r), floorf(f1.g), floorf(f1.b), floorf(f1.a));
	printf("ceilf  %+.2f, %+.2f, %+.2f, %+.2f\n", ceilf(f1.r), ceilf(f1.g), ceilf(f1.b), ceilf(f1.a));
*/
	vec4 f(-5, -0.f, 0, 5);
/*	
	printf("f = "); printv(f);
	printf("v = "); printv(v);
	
	printf("%% = "); printv(vec4(fmodf(f.x, v.x), fmodf(f.y, v.y), fmodf(f.z, v.z), fmodf(f.w, v.w)));
	printf("m = "); 
*/

//	f.xyzw.xyzw = vec4(1, 2, 3, 4);

//	f.xyzw += 1;
	
//	printv(f.xxxx.x);
	printv(f.xxxx.yyyy.zzzz.wwww);
	printv(sign(f.wzyx));

//	const vec4 t(1, 2, 3, 4);
//	printv(t.xyzw);

	return 0;
}
