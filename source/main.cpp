#include "tests/vec4.h"

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

void printv(const vec4 &v)
{
	printf("%f, %f, %f, %f\n", v.x, v.y, v.z, v.w);
}

int main()
{
	tests::vec4::testEquality();
	tests::vec4::testAccessors();
	tests::vec4::testSwizzleEquality();
	tests::vec4::testSwizzleWrite();
	tests::vec4::testUnary();

	vec4 v(4,3,2,1);
	vec4 res(1, 2, 3, 4);
	
	v.xyzw = res;
	assert(v == res);

	res = vec4(5, 6, 7, 8);
	assert(res == vec4(5, 6, 7, 8));
	
	v.wzyx = vec4(0.f, -0.f, 1.f, -1.f);
	assert(v == vec4(-1.f, 1, -0.f, 0.f));

	v.wzyx = res.xyxy;
	assert(v == vec4(6, 5, 6, 5));

		// const correctness
	const vec4 c(1, 2, 3, 4);
	assert(c == vec4(1, 2, 3, 4));
	assert(c.xxxx == vec4(1, 1, 1, 1));
	assert(c.xxxx.x == 1.0f);
	assert(c.wyzx == vec4(4, 2, 3, 1));
	assert(c.wyzx.xxxx == vec4(4, 4, 4, 4));
	assert(c.wyzx.xzyw == c.wzyx);
	assert(c.wyzx.wyzx == c);
	assert(c.wyzx.wyzx.wyzx == c.wyzx);
	assert(c.wyzx.wyzx.wyzx.wyzx == c);
	assert(c.wyzx.wyzx.wyzx.wyzx.x == c.x);
	
	return 0;
}
