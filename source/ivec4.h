#ifndef __IVEC4_H__
#define __IVEC4_H__

#include <stdint.h>
#include <emmintrin.h>

#include "swizzle.h"

class ivec4
{
	private:
		typedef _swizzle4_maker<int32_t, ivec4> _swzl;

		// ----------------------------------------------------------------- //

	public:
			// Empty constructor
		inline ivec4() {
			m = _mm_setzero_si128();
		}

			// Fill constructor
		explicit inline ivec4(int32_t i) {
			m = _mm_set1_epi32(i);
		}

			// 4 var init constructor
		inline ivec4(int32_t _x, int32_t _y, int32_t _z, int32_t _w) {
			m = _mm_setr_epi32(_x, _y, _z, _w);
		}

			// Integer array constructor
		inline ivec4(const int32_t* fv) {
			m = _mm_castps_si128(_mm_loadu_ps((const float*)fv));
		}

			// Copy constructor
		inline ivec4(const ivec4 &v) {
			m = v.m;
		}

			// SSE2 compatible constructor
		inline ivec4(const __m128i &_m) {
			m = _m;
		}

		// ----------------------------------------------------------------- //

			// Read-write swizzle
		template<unsigned mask>
		inline _swzl::rw<mask> shuffle4_rw() {
			return _swzl::rw<mask>(*this);
		}

			// Read-write swizzle, const, actually read only
		template<unsigned mask>
		inline _swzl::ro<mask> shuffle4_rw() const {
			return _swzl::ro<mask>(*this);
		}

			// Read-only swizzle
		template<unsigned mask>
		inline _swzl::ro<mask> shuffle4_ro() const {
			return _swzl::ro<mask>(*this);
		}

		// ----------------------------------------------------------------- //

			// Write direct access operator
		inline int32_t& operator[](int index) {
			return ((int32_t*)this)[index];
		}

			// Read direct access operator
		inline const int32_t& operator[](int index) const {
			return ((const int32_t*)this)[index];
		}

			// Cast operator
		inline operator int32_t* () {
			return (int32_t*)this;
		}

			// Const cast operator
		inline operator const int32_t* () const {
			return (const int32_t*)this;
		}

		// ----------------------------------------------------------------- //

		friend inline ivec4& operator += (ivec4 &v, int32_t i) {
			v.m = _mm_add_epi32(v.m, _mm_set1_epi32(i));
			return v;
		}

		friend inline ivec4& operator += (ivec4 &v0, const ivec4 &v1) {
			v0.m = _mm_add_epi32(v0.m, v1.m);
			return v0;
		}

		friend inline ivec4& operator -= (ivec4 &v, int32_t i) {
			v.m = _mm_sub_epi32(v.m, _mm_set1_epi32(i));
			return v;
		}

		friend inline ivec4& operator -= (ivec4 &v0, const ivec4 &v1) {
			v0.m = _mm_sub_epi32(v0.m, v1.m);
			return v0;
		}

		friend inline ivec4& operator *= (ivec4 &v, int32_t i) {
			__m128i ii = _mm_set1_epi32(i);
			v.m = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_mm_mul_epu32(v.m, ii)),
												  _mm_castsi128_ps(_mm_mul_epu32(_mm_srli_si128(v.m, 4),
																				 _mm_srli_si128(ii, 4))), 0x88));
			return v;
		}

		friend inline ivec4& operator *= (ivec4 &v0, const ivec4 &v1) {
			v0.m = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_mm_mul_epu32(v0.m, v1.m)),
												   _mm_castsi128_ps(_mm_mul_epu32(_mm_srli_si128(v0.m, 4),
																				  _mm_srli_si128(v1.m, 4))), 0x88));
			return v0;
		}
/*
		friend inline ivec4& operator /= (ivec4 &v, int32_t f) {
			???
		}

		friend inline ivec4& operator /= (iivec4 &v0, const ivec4 &v1) {
			???
		}
*/
		// ----------------------------------------------------------------- //

		friend inline const ivec4 operator + (int32_t i, const ivec4 &v) {
			return _mm_add_epi32(_mm_set1_epi32(i), v.m);
		}

		friend inline const ivec4 operator + (const ivec4 &v, int32_t i) {
			return _mm_add_epi32(v.m, _mm_set1_epi32(i));
		}

		friend inline const ivec4 operator + (const ivec4 &v0, const ivec4 &v1) {
			return _mm_add_epi32(v0.m, v1.m);
		}

		friend inline const ivec4 operator - (const ivec4 &v) {
			return _mm_sub_epi32(_mm_setzero_si128(), v.m);
		}

		friend inline const ivec4 operator - (int32_t i, const ivec4 &v) {
			return _mm_sub_epi32(_mm_set1_epi32(i), v.m);
		}

		friend inline const ivec4 operator - (const ivec4 &v, int32_t i) {
			return _mm_sub_epi32(v.m, _mm_set1_epi32(i));
		}

		friend inline const ivec4 operator - (const ivec4 &v0, const ivec4 &v1) {
			return _mm_sub_epi32(v0.m, v1.m);
		}

		friend inline const ivec4 operator * (int32_t i, const ivec4 &v) {
			__m128i ii = _mm_set1_epi32(i);
			return _mm_castps_si128(_mm_shuffle_ps(
									_mm_castsi128_ps(_mm_mul_epu32(ii, v.m)),
									_mm_castsi128_ps(_mm_mul_epu32(_mm_srli_si128(ii, 4),
													 _mm_srli_si128(v.m, 4))), 0x88));
		}

		friend inline const ivec4 operator * (const ivec4 &v, int32_t i) {
			__m128i ii = _mm_set1_epi32(i);
			return _mm_castps_si128(_mm_shuffle_ps(
									_mm_castsi128_ps(_mm_mul_epu32(v.m, ii)),
									_mm_castsi128_ps(_mm_mul_epu32(_mm_srli_si128(v.m, 4),
													 _mm_srli_si128(ii, 4))), 0x88));
		}

		friend inline const ivec4 operator * (const ivec4 &v0, const ivec4 &v1) {
			return _mm_castps_si128(_mm_shuffle_ps(
									_mm_castsi128_ps(_mm_mul_epu32(v0.m, v1.m)),
									_mm_castsi128_ps(_mm_mul_epu32(_mm_srli_si128(v0.m, 4),
													 _mm_srli_si128(v1.m, 4))), 0x88));
		}
/*
		friend inline const ivec4 operator / (int32_t f, const ivec4 &v) {
			???
		}

		friend inline const ivec4 operator / (const ivec4 &v, int32_t f) {
			???
		}

		friend inline const ivec4 operator / (const ivec4 &v0, const ivec4 &v1) {
			???
		}
*/
		// ----------------------------------------------------------------- //

		friend inline const ivec4 abs(const ivec4 &v) {
			__m128i mask = _mm_srai_epi32(v.m, 31);
			return _mm_xor_si128(_mm_add_epi32(v.m, mask), mask);
		}

		friend inline const ivec4 clamp(const ivec4 &v, int32_t f1, int32_t f2) {
			return max(min(v, f2), f1);
		}

		friend inline const ivec4 clamp(const ivec4 &v0,
										const ivec4 &v1, const ivec4 &v2) {
			return max(v1, min(v2, v0));
		}

		friend inline const ivec4 max(const ivec4 &v, int32_t i) {
			__m128i ii = _mm_set1_epi32(i);
			__m128i m = _mm_cmplt_epi32(v.m, ii);
	        return _mm_or_si128(_mm_andnot_si128(m, v.m), _mm_and_si128(ii, m));
		}

		friend inline const ivec4 max(const ivec4 &v0, const ivec4 &v1) {
			__m128i m = _mm_cmplt_epi32(v0.m, v1.m);
	        return _mm_or_si128(_mm_andnot_si128(m, v0.m), _mm_and_si128(v1.m, m));
		}

		friend inline const ivec4 min(const ivec4 &v, int32_t i) {
			__m128i ii = _mm_set1_epi32(i);
			__m128i m = _mm_cmpgt_epi32(v.m, ii);
	        return _mm_or_si128(_mm_andnot_si128(m, v.m), _mm_and_si128(ii, m));
		}

		friend inline const ivec4 min(const ivec4 &v0, const ivec4 &v1) {
			__m128i m = _mm_cmpgt_epi32(v0.m, v1.m);
	        return _mm_or_si128(_mm_andnot_si128(m, v0.m), _mm_and_si128(v1.m, m));
		}

		friend inline const ivec4 sign(const ivec4 &v) {
			return _mm_or_si128(_mm_add_epi32(_mm_cmpeq_epi32(v.m, _mm_setzero_si128()),
											  _mm_set1_epi32(1)),_mm_srai_epi32(v.m, 31));
		}

		// ----------------------------------------------------------------- //

		friend inline bool operator == (const ivec4 &v0, const ivec4 &v1) {
			return (_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(v0.m, v1.m))) == 0xF);
		}

		friend inline bool operator != (const ivec4 &v0, const ivec4 &v1) {
			return (_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(v0.m, v1.m))) != 0xF);
		}

		// ----------------------------------------------------------------- //

		union {
				// Vertex / Vector
			struct {
				int32_t x, y, z, w;
			};
				// Color
			struct {
				int32_t r, g, b, a;
			};
				// Texture coordinates
			struct {
				int32_t s, t, p, q;
			};

				// SSE2 register
			__m128i	m;
		};
};

#endif
