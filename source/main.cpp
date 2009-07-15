#include "tests/vec4.h"

#include <stdio.h>

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
	tests::vec4::testEquality();
	tests::vec4::testSwizzleEquality();
	tests::vec4::testAccessors();

	printf("All tests passed.\n");

	return 0;
}
