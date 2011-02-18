#include "tests/vec4.h"

#include <cmath>
#include <limits>
#include <stdio.h>
#include <assert.h>

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

#include "vec4.h"
#include "dvec4.h"
#include "ivec4.h"
#include "uvec4.h"
#include "bvec4.h"

void printv(const vec4 &v)
{
	printf("%f, %f, %f, %f\n", v.x, v.y, v.z, v.w);
}

void printv(const dvec4 &v)
{
	printf("%f, %f, %f, %f\n", v.x, v.y, v.z, v.w);
}

void printv(const ivec4 &v)
{
	printf("%d, %d, %d, %d\n", v.x, v.y, v.z, v.w);
}

void printv(const uvec4 &v)
{
	printf("%u, %u, %u, %u\n", v.x, v.y, v.z, v.w);
}

void printv(const bvec4 &v)
{
	printf("%s, %s, %s, %s\n",
		v.x ? "true" : "false",
		v.y ? "true" : "false",
		v.z ? "true" : "false",
		v.w ? "true" : "false");
}

int main()
{
	tests::vec4::testEquality();
	tests::vec4::testAccessors();
	tests::vec4::testSwizzleEquality();
	tests::vec4::testSwizzleWrite();
	tests::vec4::testUnary();

#define VECTOR dvec4

	VECTOR v  (4, 3, 2, 1);
	VECTOR res(1, 2, 3, 4);

	v.xyzw = res.wzyx;
	assert(v == VECTOR(4, 3, 2, 1));

	v.xyzw = res.wzyx.xxyy;
	assert(v == VECTOR(4, 4, 3, 3));

	v.xyzw = res;
	assert(v == res);

	res = VECTOR(5, 6, 7, 8);
	assert(res == VECTOR(5, 6, 7, 8));

	v.wzyx = VECTOR(0, 2, 1, 3);
	assert(v == VECTOR(3, 1, 2, 0));

	v.wzyx = res.xyxy;
	assert(v == VECTOR(6, 5, 6, 5));

	v.wzyx = res.xxxx.yyyy;
	assert(v == VECTOR(5, 5, 5, 5));

	v.wzyx.wzyx = VECTOR(1, 2, 3, 4);
	assert(v == VECTOR(1, 2, 3, 4));

	v.wzyx.wzyx = v.xxxx;
	assert(v == VECTOR(1, 1, 1, 1));

	v.xyzw.xyzw = VECTOR(1, 2, 3, 4);
	assert(v == VECTOR(1, 2, 3, 4));

	v.wzyx.wzyx = v.xyzw.xxxx;
	assert(v == VECTOR(1, 1, 1, 1));

	res.wzyx.y += 1;
	assert(res == VECTOR(5, 6, 8, 8));

	res.wzyx[0] += 1;
	assert(res == VECTOR(5, 6, 8, 9));

	res.wzyx += res.xyxy;
	assert(res == VECTOR(11, 11, 14, 14));

	res = clamp(res.zxwy, 12, 13);
	assert(res == VECTOR(13, 12, 13, 12));

	res = res.xxxx + res.xyzw;
	assert(res == VECTOR(26, 25, 26, 25));

	res = res.xyzw + res.zwxy;
	assert(res == VECTOR(52, 50, 52, 50));

		// const correctness
	const VECTOR c(1, 2, 3, 4);
	assert(c == VECTOR(1, 2, 3, 4));
	assert(c.xxxx == VECTOR(1, 1, 1, 1));
	assert(c.xxxx.x == 1);
	assert(c.xyzw == c.xyzw);
	assert(c.wyzx == VECTOR(4, 2, 3, 1));
	assert(c.wyzx.xxxx == VECTOR(4, 4, 4, 4));
	assert(c.wyzx.xzyw == c.wzyx);
	assert(c.wyzx.wyzx == c);
	assert(c.wyzx.wyzx.wyzx == c.wyzx);
	assert(c.wyzx.wyzx.wyzx.wyzx == c);
	assert(c.wyzx.wyzx.wyzx.wyzx.x == c.x);

	printf("All tests passed\n");

	return 0;
}
