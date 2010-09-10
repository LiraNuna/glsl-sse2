#ifndef __IVEC4_H__
#define __IVEC4_H__

#include <emmintrin.h>

class ivec4
{
	private:
			// Merges mask `target` with `m` into one unified mask that does the same sequential shuffle
		template <unsigned target, unsigned m>
		struct _mask_merger
		{
			enum
			{
				ROW0 = ((target >> (((m >> 0) & 3) << 1)) & 3) << 0,
				ROW1 = ((target >> (((m >> 2) & 3) << 1)) & 3) << 2,
				ROW2 = ((target >> (((m >> 4) & 3) << 1)) & 3) << 4,
				ROW3 = ((target >> (((m >> 6) & 3) << 1)) & 3) << 6,

				MASK = ROW0 | ROW1 | ROW2 | ROW3,
			};

			private:
				_mask_merger();
		};

			// Since we are working in little endian land, this reverses the shuffle mask
		template <unsigned m>
		struct _mask_reverser
		{
			enum
			{
				ROW0 = 0 << (((m >> 0) & 3) * 2),
				ROW1 = 1 << (((m >> 2) & 3) * 2),
				ROW2 = 2 << (((m >> 4) & 3) * 2),
				ROW3 = 3 << (((m >> 6) & 3) * 2),

				MASK = ROW0 | ROW1 | ROW2 | ROW3,
			};

			private:
				_mask_reverser();
		};

			// Swizzle helper (Read only)
		template <unsigned mask>
		struct _swzl_ro
		{
			friend class ivec4;

			public:
				inline operator const ivec4 () const {
					return _mm_shuffle_epi32(v.m, mask);
				}

					// Swizzle of the swizzle, read only const
				template<unsigned other_mask>
				inline _swzl_ro<_mask_merger<mask, other_mask>::MASK> shuffle4_ro() const {
					typedef _mask_merger<mask, other_mask> merged;
					return _swzl_ro<merged::MASK>(v);
				}

					// Swizzle of the swizzle, read/write const
				template<unsigned other_mask>
				inline _swzl_ro<_mask_merger<mask, other_mask>::MASK> shuffle4_rw() const {
					typedef _mask_merger<mask, other_mask> merged;
					return _swzl_ro<merged::MASK>(v);
				}

				const int32_t &x, &y, &z, &w;
				const int32_t &r, &g, &b, &a;
				const int32_t &s, &t, &p, &q;

			private:
					// This massive constructor maps a vector to references
				inline _swzl_ro(const ivec4 &v):
					x(v[(mask >> 0) & 0x3]), y(v[(mask >> 2) & 0x3]),
					z(v[(mask >> 4) & 0x3]), w(v[(mask >> 6) & 0x3]),

					r(v[(mask >> 0) & 0x3]), g(v[(mask >> 2) & 0x3]),
					b(v[(mask >> 4) & 0x3]), a(v[(mask >> 6) & 0x3]),

					s(v[(mask >> 0) & 0x3]), t(v[(mask >> 2) & 0x3]),
					p(v[(mask >> 4) & 0x3]), q(v[(mask >> 6) & 0x3]),

					v(v) {
						// Empty
				}

					// Reference to unswizzled self
				const ivec4 &v;
		};

			// Swizzle helper (Read/Write)
		template <unsigned mask>
		struct _swzl_rw
		{
			friend class ivec4;

			public:
				inline operator const ivec4 () const {
					return _mm_shuffle_epi32(v.m, mask);
				}

					// Swizzle from ivec4
				inline _swzl_rw& operator = (const ivec4 &r) {
					v.m = _mm_shuffle_epi32(r.m, _mask_reverser<mask>::MASK);
					return *this;
				}

					// Swizzle from same r/o mask (v1.xyzw = v2.xyzw)
				inline _swzl_rw& operator = (const _swzl_ro<mask> &s) {
					v.m = s.v.m;
					return *this;
				}

					// Swizzle from same mask (v1.xyzw = v2.xyzw)
				inline _swzl_rw& operator = (const _swzl_rw &s) {
					v.m = s.v.m;
					return *this;
				}

					// Swizzle mask => other_mask, r/o (v1.zwxy = v2.xyxy)
				template<unsigned other_mask>
				inline _swzl_rw& operator = (const _swzl_ro<other_mask> &s) {
					typedef _mask_merger<other_mask, _mask_reverser<mask>::MASK> merged;
					v.m = _mm_shuffle_epi32(s.v.m, merged::MASK);
					return *this;
				}

					// Swizzle mask => other_mask (v1.zwxy = v2.xyxy)
				template<unsigned other_mask>
				inline _swzl_rw& operator = (const _swzl_rw<other_mask> &s) {
					typedef _mask_merger<other_mask, _mask_reverser<mask>::MASK> merged;
					v.m = _mm_shuffle_epi32(s.v.m, merged::MASK);
					return *this;
				}

					// Swizzle of the swizzle, read only (v.xxxx.yyyy)
				template<unsigned other_mask>
				inline _swzl_ro<_mask_merger<mask, other_mask>::MASK> shuffle4_ro() const {
					typedef _mask_merger<mask, other_mask> merged;
					return _swzl_ro<merged::MASK>(v);
				}

					// Swizzle of the swizzle, read/write (v1.zyxw.wzyx = ...)
				template<unsigned other_mask>
				inline _swzl_rw<_mask_merger<mask, other_mask>::MASK> shuffle4_rw() {
					typedef _mask_merger<mask, other_mask> merged;
					return _swzl_rw<merged::MASK>(v);
				}

				int32_t &x, &y, &z, &w;
				int32_t &r, &g, &b, &a;
				int32_t &s, &t, &p, &q;

			private:
					// This massive contructor maps a vector to references
				inline _swzl_rw(ivec4 &v):
					x(v[(mask >> 0) & 0x3]), y(v[(mask >> 2) & 0x3]),
					z(v[(mask >> 4) & 0x3]), w(v[(mask >> 6) & 0x3]),

					r(v[(mask >> 0) & 0x3]), g(v[(mask >> 2) & 0x3]),
					b(v[(mask >> 4) & 0x3]), a(v[(mask >> 6) & 0x3]),

					s(v[(mask >> 0) & 0x3]), t(v[(mask >> 2) & 0x3]),
					p(v[(mask >> 4) & 0x3]), q(v[(mask >> 6) & 0x3]),

					v(v) {
						// Empty
				}

					// Refrence to unswizzled self
				ivec4 &v;
		};

		// ----------------------------------------------------------------- //

	public:
			// Empty constructor
		inline ivec4() {
			m = _mm_castps_si128(_mm_setzero_ps());
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
		inline _swzl_rw<mask> shuffle4_rw() {
			return _swzl_rw<mask>(*this);
		}

			// Read-write swizzle, const, actually read only
		template<unsigned mask>
		inline _swzl_ro<mask> shuffle4_rw() const {
			return _swzl_ro<mask>(*this);
		}

			// Read-only swizzle
		template<unsigned mask>
		inline _swzl_ro<mask> shuffle4_ro() const {
			return _swzl_ro<mask>(*this);
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
			v *= _mm_set1_epi32(i);
			return v;
		}

		friend inline ivec4& operator *= (ivec4 &v0, const ivec4 &v1) {
		    v0.m = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_mm_mul_epu32(v0.m, v1.m)),
												   _mm_castsi128_ps(_mm_mul_epu32(_mm_srli_si128(v0.m, 4),
																			      _mm_srli_si128(v1.m, 4))), 0x88));
		    return v0;
		}
/*
		friend inline ivec4& operator /= (ivec4 &v, float f) {
			???
		}

		friend inline ivec4& operator /= (iivec4 &v0, const ivec4 &v1) {
			???
		}
*/
		// ----------------------------------------------------------------- //

		friend inline const ivec4 operator + (const ivec4 &v, int32_t i) {
			return v + _mm_set1_epi32(i);
		}

		friend inline const ivec4 operator + (const ivec4 &v0, const ivec4 &v1) {
			return _mm_add_epi32(v0.m, v1.m);
		}

		friend inline const ivec4 operator - (const ivec4 &v) {
			return _mm_sub_epi32(_mm_set1_epi32(0), v.m);
		}

		friend inline const ivec4 operator - (int32_t i, const ivec4 &v) {
			return _mm_set1_epi32(i) - v;
		}

		friend inline const ivec4 operator - (const ivec4 &v, int32_t i) {
			return v - _mm_set1_epi32(i);
		}

		friend inline const ivec4 operator - (const ivec4 &v0, const ivec4 &v1) {
			return _mm_sub_epi32(v0.m, v1.m);
		}

		friend inline const ivec4 operator * (const ivec4 &v, int32_t i) {
		    return v * _mm_set1_epi32(i);
		}

		friend inline const ivec4 operator * (const ivec4 &v0, const ivec4 &v1) {
		    return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_mm_mul_epu32(v0.m, v1.m)),
												   _mm_castsi128_ps(_mm_mul_epu32(_mm_srli_si128(v0.m, 4),
																			      _mm_srli_si128(v1.m, 4))), 0x88));
		}
/*
		friend inline const ivec4 operator / (float f, const ivec4 &v) {
			???
		}

		friend inline const ivec4 operator / (const ivec4 &v, float f) {
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

		friend inline const ivec4 clamp(const ivec4 &v, float f1, float f2) {
			return max(min(v, f2), f1);
		}

		friend inline const ivec4 clamp(const ivec4 &v0,
										const ivec4 &v1, const ivec4 &v2) {
			return max(v1, min(v2, v0));
		}

		friend inline const ivec4 max(const ivec4 &v, int32_t i) {
			return max(v, _mm_set1_epi32(i));
		}

		friend inline const ivec4 max(const ivec4 &v0, const ivec4 &v1) {
			return _mm_xor_si128(v0.m, _mm_and_si128(_mm_xor_si128(v0.m, v1.m),
													 _mm_cmplt_epi32(v0.m, v1.m)));
		}

		friend inline const ivec4 min(const ivec4 &v, int32_t i) {
			return min(v, _mm_set1_epi32(i));
		}

		friend inline const ivec4 min(const ivec4 &v0, const ivec4 &v1) {
			return _mm_xor_si128(v1.m, _mm_and_si128(_mm_xor_si128(v0.m, v1.m),
													 _mm_cmplt_epi32(v0.m, v1.m)));
		}

		friend inline const ivec4 sign(const ivec4 &v) {
			return _mm_or_si128(_mm_add_epi32(_mm_cmpeq_epi32(v.m, _mm_set1_epi32(0)),
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

#include "swizzle.h"

#endif
