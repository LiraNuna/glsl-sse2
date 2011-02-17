#ifndef __UVEC4_H__
#define __UVEC4_H__

#include <stdint.h>
#include <emmintrin.h>

#include "swizzle.h"

class uvec4
{
	private:
		typedef _swizzle4_maker<uint32_t, uvec4> _swzl;

		// ----------------------------------------------------------------- //

	public:
			// Empty constructor
		inline uvec4() {
			m = _mm_setzero_si128();
		}

			// Fill constructor
		explicit inline uvec4(uint32_t i) {
			m = _mm_set1_epi32(i);
		}

			// 4 var init constructor
		inline uvec4(uint32_t _x, uint32_t _y, uint32_t _z, uint32_t _w) {
			m = _mm_setr_epi32(_x, _y, _z, _w);
		}

			// Integer array constructor
		inline uvec4(const uint32_t* fv) {
			m = _mm_castps_si128(_mm_loadu_ps((const float*)fv));
		}

			// Copy constructor
		inline uvec4(const uvec4 &v) {
			m = v.m;
		}

			// SSE2 compatible constructor
		inline uvec4(const __m128i &_m) {
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
		inline uint32_t& operator[](int index) {
			return ((uint32_t*)this)[index];
		}

			// Read direct access operator
		inline const uint32_t& operator[](int index) const {
			return ((const uint32_t*)this)[index];
		}

			// Cast operator
		inline operator uint32_t* () {
			return (uint32_t*)this;
		}

			// Const cast operator
		inline operator const uint32_t* () const {
			return (const uint32_t*)this;
		}

		// ----------------------------------------------------------------- //

		friend inline uvec4& operator += (uvec4 &v, uint32_t u) {
			v.m = _mm_add_epi32(v.m, _mm_set1_epi32(u));
			return v;
		}

		friend inline uvec4& operator += (uvec4 &v0, const uvec4 &v1) {
			v0.m = _mm_add_epi32(v0.m, v1.m);
			return v0;
		}

		friend inline uvec4& operator -= (uvec4 &v, uint32_t u) {
			v.m = _mm_sub_epi32(v.m, _mm_set1_epi32(u));
			return v;
		}

		friend inline uvec4& operator -= (uvec4 &v0, const uvec4 &v1) {
			v0.m = _mm_sub_epi32(v0.m, v1.m);
			return v0;
		}

		friend inline uvec4& operator *= (uvec4 &v, uint32_t u) {
			__m128i uu = _mm_set1_epi32(u);
			v.m = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_mm_mul_epu32(v.m, uu)),
												  _mm_castsi128_ps(_mm_mul_epu32(_mm_srli_si128(v.m, 4),
																				 _mm_srli_si128(uu, 4))), 0x88));
			return v;
		}

		friend inline uvec4& operator *= (uvec4 &v0, const uvec4 &v1) {
			v0.m = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_mm_mul_epu32(v0.m, v1.m)),
												   _mm_castsi128_ps(_mm_mul_epu32(_mm_srli_si128(v0.m, 4),
																				  _mm_srli_si128(v1.m, 4))), 0x88));
			return v0;
		}
/*
		friend inline uvec4& operator /= (uvec4 &v, float f) {
			???
		}

		friend inline uvec4& operator /= (iuvec4 &v0, const uvec4 &v1) {
			???
		}
*/
		// ----------------------------------------------------------------- //

		friend inline const uvec4 operator + (uint32_t i, const uvec4 &v) {
			return _mm_add_epi32(_mm_set1_epi32(i), v.m);
		}

		friend inline const uvec4 operator + (const uvec4 &v, uint32_t i) {
			return _mm_add_epi32(v.m, _mm_set1_epi32(i));
		}

		friend inline const uvec4 operator + (const uvec4 &v0, const uvec4 &v1) {
			return _mm_add_epi32(v0.m, v1.m);
		}

		friend inline const uvec4 operator - (uint32_t u, const uvec4 &v) {
			return _mm_sub_epi32(_mm_set1_epi32(u), v.m);
		}

		friend inline const uvec4 operator - (const uvec4 &v, uint32_t u) {
			return _mm_sub_epi32(v.m, _mm_set1_epi32(u));
		}

		friend inline const uvec4 operator - (const uvec4 &v0, const uvec4 &v1) {
			return _mm_sub_epi32(v0.m, v1.m);
		}

		friend inline const uvec4 operator * (uint32_t u, const uvec4 &v) {
			__m128i uu = _mm_set1_epi32(u);
			return _mm_castps_si128(_mm_shuffle_ps(
									_mm_castsi128_ps(_mm_mul_epu32(uu, v.m)),
									_mm_castsi128_ps(_mm_mul_epu32(_mm_srli_si128(uu, 4),
													 _mm_srli_si128(v.m, 4))), 0x88));
		}

		friend inline const uvec4 operator * (const uvec4 &v, uint32_t u) {
			__m128i uu = _mm_set1_epi32(u);
			return _mm_castps_si128(_mm_shuffle_ps(
									_mm_castsi128_ps(_mm_mul_epu32(v.m, uu)),
									_mm_castsi128_ps(_mm_mul_epu32(_mm_srli_si128(v.m, 4),
													 _mm_srli_si128(uu, 4))), 0x88));
		}

		friend inline const uvec4 operator * (const uvec4 &v0, const uvec4 &v1) {
			return _mm_castps_si128(_mm_shuffle_ps(
									_mm_castsi128_ps(_mm_mul_epu32(v0.m, v1.m)),
									_mm_castsi128_ps(_mm_mul_epu32(_mm_srli_si128(v0.m, 4),
													 _mm_srli_si128(v1.m, 4))), 0x88));
		}
/*
		friend inline const uvec4 operator / (uint32_t u, const uvec4 &v) {
			???
		}

		friend inline const uvec4 operator / (const uvec4 &v, uint32_t u) {
			???
		}

		friend inline const uvec4 operator / (const uvec4 &v0, const uvec4 &v1) {
			???
		}
*/
		// ----------------------------------------------------------------- //

		friend inline const uvec4 clamp(const uvec4 &v, uint32_t u1, uint32_t u2) {
			return max(min(v, u2), u1);
		}

		friend inline const uvec4 clamp(const uvec4 &v0,
										const uvec4 &v1, const uvec4 &v2) {
			return max(v1, min(v2, v0));
		}

		friend inline const uvec4 max(const uvec4 &v, uint32_t u) {
			__m128i uu = _mm_set1_epi32(u);
			__m128i m = _mm_set1_epi32(0x80000000);
			__m128i mm = _mm_cmplt_epi32(_mm_xor_si128(v.m, m), _mm_xor_si128(uu, m));
	        return _mm_or_si128(_mm_andnot_si128(mm, v.m), _mm_and_si128(uu, mm));
		}

		friend inline const uvec4 max(const uvec4 &v0, const uvec4 &v1) {
			__m128i m = _mm_set1_epi32(0x80000000);
			__m128i mm = _mm_cmplt_epi32(_mm_xor_si128(v0.m, m), _mm_xor_si128(v1.m, m));
	        return _mm_or_si128(_mm_andnot_si128(mm, v0.m), _mm_and_si128(v1.m, mm));
		}

		friend inline const uvec4 min(const uvec4 &v, uint32_t u) {
			__m128i uu = _mm_set1_epi32(u);
			__m128i m = _mm_set1_epi32(0x80000000);
			__m128i mm = _mm_cmpgt_epi32(_mm_xor_si128(v.m, m), _mm_xor_si128(uu, m));
	        return _mm_or_si128(_mm_andnot_si128(mm, v.m), _mm_and_si128(uu, mm));
		}

		friend inline const uvec4 min(const uvec4 &v0, const uvec4 &v1) {
			__m128i m = _mm_set1_epi32(0x80000000);
			__m128i mm = _mm_cmpgt_epi32(_mm_xor_si128(v0.m, m), _mm_xor_si128(v1.m, m));
	        return _mm_or_si128(_mm_andnot_si128(mm, v0.m), _mm_and_si128(v1.m, mm));
		}

		// ----------------------------------------------------------------- //

		friend inline bool operator == (const uvec4 &v0, const uvec4 &v1) {
			return (_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(v0.m, v1.m))) == 0xF);
		}

		friend inline bool operator != (const uvec4 &v0, const uvec4 &v1) {
			return (_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(v0.m, v1.m))) != 0xF);
		}

		// ----------------------------------------------------------------- //

		union {
				// Vertex / Vector
			struct {
				uint32_t x, y, z, w;
			};
				// Color
			struct {
				uint32_t r, g, b, a;
			};
				// Texture coordinates
			struct {
				uint32_t s, t, p, q;
			};

				// SSE2 register
			__m128i	m;
		};
};

#endif
