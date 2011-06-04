#ifndef __UVEC4_H__
#define __UVEC4_H__

#include <stdint.h>
#include <emmintrin.h>

class uvec4
{
	private:
		template<unsigned mask>
		static inline __m128i shuffle(const __m128i &xmm) {
			return _mm_shuffle_epi32(xmm, mask);
		}

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
				ROW0 = 0 << (((m >> 0) & 3) << 1),
				ROW1 = 1 << (((m >> 2) & 3) << 1),
				ROW2 = 2 << (((m >> 4) & 3) << 1),
				ROW3 = 3 << (((m >> 6) & 3) << 1),

				MASK = ROW0 | ROW1 | ROW2 | ROW3,
			};

			private:
				_mask_reverser();
		};

			// Swizzle helper (Read only)
		template <unsigned mask>
		struct _swzl_ro
		{
			friend class uvec4;

			public:
				inline operator const uvec4 () const {
					return shuffle<mask>(v.m);
				}

				inline uint32_t operator[](int index) const {
					return v[(mask >> (index << 1)) & 0x3];
				}

					// Swizzle of the swizzle, read only const
				template<unsigned other_mask>
				inline _swzl_ro<_mask_merger<mask, other_mask>::MASK> shuffle4_ro2() const {
					typedef _mask_merger<mask, other_mask> merged;
					return _swzl_ro<merged::MASK>(v);
				}

					// Swizzle of the swizzle, read only const (4)
				template<unsigned other_mask>
				inline _swzl_ro<_mask_merger<mask, other_mask>::MASK> shuffle4_ro4() const {
					typedef _mask_merger<mask, other_mask> merged;
					return _swzl_ro<merged::MASK>(v);
				}

					// Swizzle of the swizzle, read/write const
				template<unsigned other_mask>
				inline _swzl_ro<_mask_merger<mask, other_mask>::MASK> shuffle4_rw4() const {
					typedef _mask_merger<mask, other_mask> merged;
					return _swzl_ro<merged::MASK>(v);
				}

				const uint32_t &x, &y, &z, &w;
				const uint32_t &r, &g, &b, &a;
				const uint32_t &s, &t, &p, &q;

			private:
					// This massive constructor maps a vector to references
				inline _swzl_ro(const uvec4 &v):
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
				const uvec4 &v;
		};

			// Swizzle helper (Read/Write)
		template <unsigned mask>
		struct _swzl_rw
		{
			friend class uvec4;

			public:
				inline operator const uvec4 () const {
					return shuffle<mask>(v.m);
				}

				inline uint32_t& operator[](int index) const {
					return v[(mask >> (index << 1)) & 0x3];
				}

					// Swizzle from uvec4
				inline uvec4& operator = (const uvec4 &r) {
					return v = shuffle<_mask_reverser<mask>::MASK>(r.m);
				}

					// Swizzle from same r/o mask (v1.xyzw = v2.xyzw)
				inline uvec4& operator = (const _swzl_ro<mask> &s) {
					return v = s.v;
				}

					// Swizzle from same mask (v1.xyzw = v2.xyzw)
				inline uvec4& operator = (const _swzl_rw &s) {
					return v = s.v;
				}

					// Swizzle mask => other_mask, r/o (v1.zwxy = v2.xyxy)
				template<unsigned other_mask>
				inline uvec4& operator = (const _swzl_ro<other_mask> &s) {
					typedef _mask_merger<other_mask, _mask_reverser<mask>::MASK> merged;

					return v = shuffle<merged::MASK>(s.v.m);
				}

					// Swizzle mask => other_mask (v1.zwxy = v2.xyxy)
				template<unsigned other_mask>
				inline uvec4& operator = (const _swzl_rw<other_mask> &s) {
					typedef _mask_merger<other_mask, _mask_reverser<mask>::MASK> merged;

					return v = shuffle<merged::MASK>(s.v.m);
				}

					// Swizzle of the swizzle, read only (v.xxxx.yyyy) (2)
				template<unsigned other_mask>
				inline _swzl_ro<_mask_merger<mask, other_mask>::MASK> shuffle4_ro2() const {
					typedef _mask_merger<mask, other_mask> merged;

					return _swzl_ro<merged::MASK>(v);
				}

					// Swizzle of the swizzle, read only (v.xxxx.yyyy) (4)
				template<unsigned other_mask>
				inline _swzl_ro<_mask_merger<mask, other_mask>::MASK> shuffle4_ro4() const {
					typedef _mask_merger<mask, other_mask> merged;

					return _swzl_ro<merged::MASK>(v);
				}

					// Swizzle of the swizzle, read/write (v1.zyxw.wzyx = ...)
				template<unsigned other_mask>
				inline _swzl_rw<_mask_merger<mask, other_mask>::MASK> shuffle4_rw4() {
					typedef _mask_merger<mask, other_mask> merged;

					return _swzl_rw<merged::MASK>(v);
				}

				// ----------------------------------------------------------------- //

				inline uvec4& operator += (uint32_t s) {
					return v += s;
				}

				inline uvec4& operator += (const uvec4 &v0) {
					return v += v0.shuffle4_ro4<mask>();
				}

				inline uvec4& operator -= (uint32_t s) {
					return v -= s;
				}

				inline uvec4& operator -= (const uvec4 &v0) {
					return v -= v0.shuffle4_ro4<mask>();
				}

				inline uvec4& operator *= (uint32_t s) {
					return v *= s;
				}

				inline uvec4& operator *= (const uvec4 &v0) {
					return v *= v0.shuffle4_ro4<mask>();
				}

				inline uvec4& operator /= (uint32_t s) {
					return v /= s;
				}

				inline uvec4& operator /= (const uvec4 &v0) {
					return v /= v0.shuffle4_ro4<mask>();
				}

				// ----------------------------------------------------------------- //

				uint32_t &x, &y, &z, &w;
				uint32_t &r, &g, &b, &a;
				uint32_t &s, &t, &p, &q;

			private:
					// This massive contructor maps a vector to references
				inline _swzl_rw(uvec4 &v):
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
				uvec4 &v;
		};

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

		inline void* operator new(size_t size) throw() {
			return _mm_malloc(size, 16);
		}

		inline void operator delete(void* ptr) {
			_mm_free(ptr);
		}

		// ----------------------------------------------------------------- //

			// Read-write swizzle
		template<unsigned mask>
		inline _swzl_rw<mask> shuffle4_rw4() {
			return _swzl_rw<mask>(*this);
		}

			// Read-write swizzle, const, actually read only
		template<unsigned mask>
		inline _swzl_ro<mask> shuffle4_rw4() const {
			return _swzl_ro<mask>(*this);
		}

			// Read-only swizzle (2)
		template<unsigned mask>
		inline _swzl_ro<mask> shuffle4_ro2() const {
			return _swzl_ro<mask>(*this);
		}

			// Read-only swizzle (4)
		template<unsigned mask>
		inline _swzl_ro<mask> shuffle4_ro4() const {
			return _swzl_ro<mask>(*this);
		}

		// ----------------------------------------------------------------- //

			// Write direct access operator
		inline uint32_t& operator[](int index) {
			return reinterpret_cast<uint32_t *>(this)[index];
		}

			// Read direct access operator
		inline const uint32_t& operator[](int index) const {
			return reinterpret_cast<const uint32_t *>(this)[index];
		}

			// Cast operator
		inline operator uint32_t* () {
			return reinterpret_cast<uint32_t *>(this);
		}

			// Const cast operator
		inline operator const uint32_t* () const {
			return reinterpret_cast<const uint32_t *>(this);
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
		friend inline uvec4& operator /= (uvec4 &v, uint32_t f) {
			// TODO
		}

		friend inline uvec4& operator /= (iuvec4 &v0, const uvec4 &v1) {
			// TODO
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
			// TODO
		}

		friend inline const uvec4 operator / (const uvec4 &v, uint32_t u) {
			// TODO
		}

		friend inline const uvec4 operator / (const uvec4 &v0, const uvec4 &v1) {
			// TODO
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

// Template specialization for mask 0xE4 (No shuffle)
template<>
inline __m128i uvec4::shuffle<0xE4>(const __m128i &xmm) {
	return xmm;
}

#include "swizzle4.h"

#endif
