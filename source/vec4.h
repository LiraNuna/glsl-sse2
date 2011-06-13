#ifndef __VEC4_H__
#define __VEC4_H__

#include <emmintrin.h>

class vec4
{
	private:
			// Most compilers don't use pshufd (SSE2) when _mm_shuffle(x, x, mask) is used
			// This macro saves 2-3 movaps instructions when shuffling
			// This has to be a macro since mask HAS to be an immidiate value
		#define _mm_shufd(xmm, mask) _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(xmm), mask))

		template<unsigned mask>
		static inline __m128 shuffle(const __m128 &xmm) {
			return _mm_shufd(xmm, mask);
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
			friend class vec4;

			public:
				inline operator const vec4 () const {
					return shuffle<mask>(v.m);
				}

				inline float operator[](int index) const {
					return v[(mask >> (index << 1)) & 0x3];
				}

					// Swizzle of the swizzle, read only const (2)
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

				const float &x, &y, &z, &w;
				const float &r, &g, &b, &a;
				const float &s, &t, &p, &q;

			private:
					// This massive constructor maps a vector to references
				inline _swzl_ro(const vec4 &v):
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
				const vec4 &v;
		};

			// Swizzle helper (Read/Write)
		template <unsigned mask>
		struct _swzl_rw
		{
			friend class vec4;

			public:
				inline operator const vec4 () const {
					return shuffle<mask>(v.m);
				}

				inline float& operator[](int index) {
					return v[(mask >> (index << 1)) & 0x3];
				}

					// Swizzle from vec4
				inline vec4& operator = (const vec4 &r) {
					return v = shuffle<_mask_reverser<mask>::MASK>(r.m);
				}

					// Swizzle from same r/o mask (v1.xyzw = v2.xyzw)
				inline vec4& operator = (const _swzl_ro<mask> &s) {
					return v = s.v;
				}

					// Swizzle from same mask (v1.xyzw = v2.xyzw)
				inline vec4& operator = (const _swzl_rw &s) {
					return v = s.v;
				}

					// Swizzle mask => other_mask, r/o (v1.zwxy = v2.xyxy)
				template<unsigned other_mask>
				inline vec4& operator = (const _swzl_ro<other_mask> &s) {
					typedef _mask_merger<other_mask, _mask_reverser<mask>::MASK> merged;

					return v = shuffle<merged::MASK>(s.v.m);
				}

					// Swizzle mask => other_mask (v1.zwxy = v2.xyxy)
				template<unsigned other_mask>
				inline vec4& operator = (const _swzl_rw<other_mask> &s) {
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

				inline vec4& operator += (float s) {
					return v += s;
				}

				inline vec4& operator += (const vec4 &v0) {
					return v += v0.shuffle4_ro4<mask>();
				}

				inline vec4& operator -= (float s) {
					return v -= s;
				}

				inline vec4& operator -= (const vec4 &v0) {
					return v -= v0.shuffle4_ro4<mask>();
				}

				inline vec4& operator *= (float s) {
					return v *= s;
				}

				inline vec4& operator *= (const vec4 &v0) {
					return v *= v0.shuffle4_ro4<mask>();
				}

				inline vec4& operator /= (float s) {
					return v /= s;
				}

				inline vec4& operator /= (const vec4 &v0) {
					return v /= v0.shuffle4_ro4<mask>();
				}

				// ----------------------------------------------------------------- //

				float &x, &y, &z, &w;
				float &r, &g, &b, &a;
				float &s, &t, &p, &q;

			private:
					// This massive contructor maps a vector to references
				inline _swzl_rw(vec4 &v):
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
				vec4 &v;
		};

		// ----------------------------------------------------------------- //

	public:
			// Empty constructor
		inline vec4() {
			m = _mm_setzero_ps();
		}

			// Fill constructor
		explicit inline vec4(float f) {
			m = _mm_set1_ps(f);
		}

			// 4 var init constructor
		inline vec4(float _x, float _y, float _z, float _w) {
			m = _mm_setr_ps(_x, _y, _z, _w);
		}

			// Float array constructor
		inline vec4(const float* fv) {
			m = _mm_loadu_ps(fv);
		}

			// Copy constructor
		inline vec4(const vec4 &v) {
			m = v.m;
		}

			// SSE compatible constructor
		inline vec4(const __m128 &_m) {
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
		inline float& operator[](int index) {
			return reinterpret_cast<float *>(this)[index];
		}

			// Read direct access operator
		inline const float& operator[](int index) const {
			return reinterpret_cast<const float *>(this)[index];
		}

			// Cast operator
		inline operator float* () {
			return reinterpret_cast<float *>(this);
		}

			// Const cast operator
		inline operator const float* () const {
			return reinterpret_cast<const float *>(this);
		}

		// ----------------------------------------------------------------- //

		friend inline vec4& operator += (vec4 &v, float f) {
			v.m = _mm_add_ps(v.m, _mm_set1_ps(f));
			return v;
		}

		friend inline vec4& operator += (vec4 &v0, const vec4 &v1) {
			v0.m = _mm_add_ps(v0.m, v1.m);
			return v0;
		}

		friend inline vec4& operator -= (vec4 &v, float f) {
			v.m = _mm_sub_ps(v.m, _mm_set1_ps(f));
			return v;
		}

		friend inline vec4& operator -= (vec4 &v0, const vec4 &v1) {
			v0.m = _mm_sub_ps(v0.m, v1.m);
			return v0;
		}

		friend inline vec4& operator *= (vec4 &v, float f) {	
			v.m = _mm_mul_ps(v.m, _mm_set1_ps(f));
			return v;
		}

		friend inline vec4& operator *= (vec4 &v0, const vec4 &v1) {
			v0.m = _mm_mul_ps(v0.m, v1.m);
			return v0;
		}

		friend inline vec4& operator /= (vec4 &v, float f) {
			v.m = _mm_div_ps(v.m, _mm_set1_ps(f));
			return v;
		}

		friend inline vec4& operator /= (vec4 &v0, const vec4 &v1) {
			v0.m = _mm_div_ps(v0.m, v1.m);
			return v0;
		}

		// ----------------------------------------------------------------- //

		friend inline const vec4 operator + (float f, const vec4 &v) {
			return _mm_add_ps(_mm_set1_ps(f), v.m);
		}

		friend inline const vec4 operator + (const vec4 &v, float f) {
			return _mm_add_ps(v.m, _mm_set1_ps(f));
		}

		friend inline const vec4 operator + (const vec4 &v0, const vec4 &v1) {
			return _mm_add_ps(v0.m, v1.m);
		}

		friend inline const vec4 operator - (const vec4 &v) {
			return _mm_xor_ps(v.m, _mm_set1_ps(-0.f));
		}

		friend inline const vec4 operator - (float f, const vec4 &v) {
			return _mm_sub_ps( _mm_set1_ps(f), v.m);
		}

		friend inline const vec4 operator - (const vec4 &v, float f) {
			return _mm_sub_ps(v.m, _mm_set1_ps(f));
		}

		friend inline const vec4 operator - (const vec4 &v0, const vec4 &v1) {
			return _mm_sub_ps(v0.m, v1.m);
		}

		friend inline const vec4 operator * (float f, const vec4 &v) {
			return _mm_mul_ps(_mm_set1_ps(f), v.m);
		}

		friend inline const vec4 operator * (const vec4 &v, float f) {
			return _mm_mul_ps(v.m, _mm_set1_ps(f));
		}

		friend inline const vec4 operator * (const vec4 &v0, const vec4 &v1) {
			return _mm_mul_ps(v0.m, v1.m);
		}

		friend inline const vec4 operator / (float f, const vec4 &v) {
			return _mm_div_ps(_mm_set1_ps(f), v.m);
		}

		friend inline const vec4 operator / (const vec4 &v, float f) {
			return _mm_div_ps(v.m, _mm_set1_ps(f));
		}

		friend inline const vec4 operator / (const vec4 &v0, const vec4 &v1) {
			return _mm_div_ps(v0.m, v1.m);
		}

		// ----------------------------------------------------------------- //

		friend inline const vec4 pow(const vec4 &v0, const vec4 &v1) {
			return exp2(log2(abs(v0)) * v1);
		}
/*
		friend inline const vec4 exp(const vec4 &v) {
			// TODO
		}
*/
		friend inline const vec4 log(const vec4 &v) {
			return log2(v) * 0.69314718055995f;
		}

		friend inline const vec4 exp2(const vec4 &v) {
			__m128i ix = _mm_cvttps_epi32(_mm_add_ps(v.m, _mm_castsi128_ps(
										  _mm_andnot_si128(_mm_srai_epi32(
														   _mm_cvttps_epi32(v.m), 31),
										  _mm_set1_epi32(0x3F7FFFFF)))));
			__m128 f = _mm_mul_ps(_mm_sub_ps(_mm_cvtepi32_ps(ix), v.m),
											 _mm_set1_ps(0.69314718055994530942f));
			__m128 hi = _mm_add_ps(_mm_mul_ps(f, _mm_set1_ps(-0.0001413161f)),
												 _mm_set1_ps( 0.0013298820f));
			__m128 lo = _mm_add_ps(_mm_mul_ps(f, _mm_set1_ps(-0.1666653019f)),
												 _mm_set1_ps( 0.4999999206f));
			hi = _mm_add_ps(_mm_mul_ps(f, hi), _mm_set1_ps(-0.0083013598f));
			hi = _mm_add_ps(_mm_mul_ps(f, hi), _mm_set1_ps( 0.0416573475f));
			lo = _mm_add_ps(_mm_mul_ps(f, lo), _mm_set1_ps(-0.9999999995f));
			lo = _mm_add_ps(_mm_mul_ps(f, lo), _mm_set1_ps(1.0f));
			__m128 f2 = _mm_mul_ps(f, f);
			return _mm_or_ps(_mm_mul_ps(_mm_add_ps(
							 _mm_mul_ps(_mm_mul_ps(f2, f2), hi), lo),
							 _mm_castsi128_ps(_mm_and_si128(_mm_slli_epi32((
									 _mm_add_epi32(ix, _mm_set1_epi32(127))), 23),
									 _mm_cmpgt_epi32(ix, _mm_set1_epi32(-128))))),
							 _mm_castsi128_ps(_mm_srli_epi32(
									 _mm_cmpgt_epi32(ix, _mm_set1_epi32( 128)), 1)));
		}

		friend inline const vec4 log2(const vec4 &v) {
			__m128i e = _mm_sub_epi32(_mm_srli_epi32(_mm_castps_si128(
									  _mm_andnot_ps(_mm_set1_ps(-0.0f), v.m)), 23),
												    _mm_set1_epi32(127));
			__m128 y = _mm_sub_ps(_mm_castsi128_ps(_mm_sub_epi32(
								  _mm_castps_si128(v.m), _mm_slli_epi32(e, 23))),
														 _mm_set1_ps(1.0f));
			__m128 x2 = _mm_mul_ps(y, y);
			__m128 x4 = _mm_mul_ps(x2, x2);
			__m128 hi = _mm_add_ps(_mm_mul_ps(y, _mm_set1_ps(-0.00931049621349f)),
												 _mm_set1_ps( 0.05206469089414f));
			__m128 lo = _mm_add_ps(_mm_mul_ps(y, _mm_set1_ps( 0.47868480909345f)),
												 _mm_set1_ps(-0.72116591947498f));
			hi = _mm_add_ps(_mm_mul_ps(y, hi),   _mm_set1_ps(-0.13753123777116f));
			hi = _mm_add_ps(_mm_mul_ps(y, hi),   _mm_set1_ps( 0.24187369696082f));
			hi = _mm_add_ps(_mm_mul_ps(y, hi),   _mm_set1_ps(-0.34730547155299f));
			lo = _mm_add_ps(_mm_mul_ps(y, lo),   _mm_set1_ps(1.442689881667200f));
			return _mm_add_ps(_mm_add_ps(_mm_mul_ps(x4, hi),
										 _mm_mul_ps(y, lo)), _mm_cvtepi32_ps(e));
		}

		friend inline const vec4 sqrt(const vec4 &v) {
			return _mm_sqrt_ps(v.m);
		}

		friend inline const vec4 inversesqrt(const vec4 &v) {
			return _mm_div_ps(_mm_set1_ps(1.0f), _mm_sqrt_ps(v.m));
		}

		// ----------------------------------------------------------------- //

		friend inline const vec4 abs(const vec4 &v) {
			return _mm_andnot_ps(_mm_set1_ps(-0.f), v.m);
		}

		friend inline const vec4 ceil(const vec4 &v) {
			__m128 m = _mm_cmpunord_ps(v.m,
					   _mm_cmpge_ps(_mm_andnot_ps(_mm_set1_ps(-0.0f), v.m),
												  _mm_set1_ps(8388608.0f)));
			return _mm_or_ps(_mm_andnot_ps(m, _mm_cvtepi32_ps(_mm_cvtps_epi32(
							 _mm_add_ps(v.m, _mm_set1_ps(0.5f))))),
											 _mm_and_ps(m, v.m));
		}

		friend inline const vec4 clamp(const vec4 &v0, float f1, float f2) {
			return _mm_max_ps(_mm_set1_ps(f1),
							  _mm_min_ps(_mm_set1_ps(f2), v0.m));
		}

		friend inline const vec4 clamp(const vec4 &v0,
									   const vec4 &v1, const vec4 &v2) {
			return _mm_max_ps(v1.m, _mm_min_ps(v2.m, v0.m));
		}

		friend inline const vec4 floor(const vec4 &v) {
			__m128 m = _mm_cmpunord_ps(v.m,
					   _mm_cmpge_ps(_mm_andnot_ps(_mm_set1_ps(-0.0f), v.m),
												  _mm_set1_ps(8388608.0f)));
			return _mm_or_ps(_mm_andnot_ps(m, _mm_cvtepi32_ps(_mm_cvtps_epi32(
							 _mm_sub_ps(v.m, _mm_set1_ps(0.5f))))),
											 _mm_and_ps(m, v.m));
		}

		friend inline const vec4 fract(const vec4 &v) {
			return _mm_sub_ps(v.m, _mm_cvtepi32_ps(_mm_cvtps_epi32(
								   _mm_sub_ps(v.m, _mm_set1_ps(0.5f)))));
		}

		friend inline const vec4 max(const vec4 &v, float f) {
			return _mm_max_ps(v.m, _mm_set1_ps(f));
		}

		friend inline const vec4 max(const vec4 &v0, const vec4 &v1) {
			return _mm_max_ps(v0.m, v1.m);
		}

		friend inline const vec4 min(const vec4 &v, float f) {
			return _mm_min_ps(v.m, _mm_set1_ps(f));
		}

		friend inline const vec4 min(const vec4 &v0, const vec4 &v1) {
			return _mm_min_ps(v0.m, v1.m);
		}

		friend inline const vec4 mix(const vec4 &v0, const vec4 &v1,
									 float f) {
			__m128 ff = _mm_set1_ps(f);
			return _mm_add_ps(_mm_mul_ps(v0.m, _mm_sub_ps(_mm_set1_ps(1.f), ff)),
							  _mm_mul_ps(v1.m, ff));
		}

		friend inline const vec4 mix(const vec4 &v0, const vec4 &v1,
									 const vec4 &v2) {
			return _mm_add_ps(_mm_mul_ps(v0.m, _mm_sub_ps(_mm_set1_ps(1.f), v1.m)),
							  _mm_mul_ps(v1.m, v2.m));
		}

		friend inline const vec4 mod(const vec4 &v, float f) {
			__m128 ff = _mm_set1_ps(f);
			__m128 d = _mm_div_ps(v.m, ff);
			__m128 m = _mm_cmpunord_ps(d,
					   _mm_cmpge_ps(_mm_andnot_ps(_mm_set1_ps(-0.0f), d),
												  _mm_set1_ps(8388608.0f)));
			return _mm_sub_ps(v.m, _mm_mul_ps(ff, _mm_or_ps(_mm_andnot_ps(m,
								   _mm_cvtepi32_ps(_mm_cvtps_epi32(
								   _mm_sub_ps(d, _mm_set1_ps(0.5f))))),
												 _mm_and_ps(m, d))));
		}

		friend inline const vec4 mod(const vec4 &v0, const vec4 &v1) {
			__m128 d = _mm_div_ps(v0.m, v1.m);
			__m128 m = _mm_cmpunord_ps(d,
					   _mm_cmpge_ps(_mm_andnot_ps(_mm_set1_ps(-0.0f), d),
												  _mm_set1_ps(8388608.0f)));
			return _mm_sub_ps(v0.m, _mm_mul_ps(v1.m, _mm_or_ps(_mm_andnot_ps(m,
									_mm_cvtepi32_ps(_mm_cvtps_epi32(
									_mm_sub_ps(d, _mm_set1_ps(0.5f))))),
												  _mm_and_ps(m, d))));
		}

		friend inline const vec4 modf(const vec4 &v0, vec4 &v1) {
			__m128 m = _mm_cmpunord_ps(v0.m,
					   _mm_cmpge_ps(_mm_andnot_ps(_mm_set1_ps(-0.0f), v0.m),
												  _mm_set1_ps(8388608.0f)));
			v1.m = _mm_or_ps(_mm_andnot_ps(m, _mm_cvtepi32_ps(_mm_cvtps_epi32(
							 _mm_sub_ps(v0.m, _mm_set1_ps(0.5f))))),
											  _mm_and_ps(m, v0.m));
			return _mm_sub_ps(v0.m, v1.m);
		}

		friend inline const vec4 round(const vec4 &v) {
			__m128 m = _mm_cmpunord_ps(v.m,
					   _mm_cmpge_ps(_mm_andnot_ps(_mm_set1_ps(-0.0f), v.m),
												  _mm_set1_ps(8388608.0f)));
			return _mm_or_ps(_mm_andnot_ps(m, _mm_cvtepi32_ps(
							 _mm_cvtps_epi32(v.m))), _mm_and_ps(m, v.m));
		}

		friend inline const vec4 roundEven(const vec4 &v) {
			__m128 m = _mm_or_ps(_mm_and_ps(_mm_set1_ps(-0.0f), v.m),
			                                _mm_set1_ps(8388608.0f));
			return _mm_sub_ps(_mm_add_ps(v.m, m), m);
		}

		friend inline const vec4 sign(const vec4 &v) {
			return _mm_and_ps(_mm_or_ps(_mm_and_ps(v.m, _mm_set1_ps(-0.f)),
										_mm_set1_ps(1.0f)),
							  _mm_cmpneq_ps(v.m, _mm_setzero_ps()));
		}

		friend inline const vec4 smoothstep(float f1, float f2,
											const vec4 &v) {
			__m128 ff1 = _mm_set1_ps(f1);
			 __m128 c = _mm_max_ps(_mm_min_ps(_mm_div_ps(_mm_sub_ps(v.m, ff1),
								   _mm_sub_ps(_mm_set1_ps(f2), ff1)),
								   _mm_set1_ps(1.f)), _mm_setzero_ps());
			return _mm_mul_ps(_mm_mul_ps(c, c),
							  _mm_sub_ps(_mm_set1_ps(3.0f), _mm_add_ps(c, c)));
		}

		friend inline const vec4 smoothstep(const vec4 &v0,
											const vec4 &v1, const vec4 &v2) {
			 __m128 c = _mm_max_ps(_mm_min_ps(_mm_div_ps(_mm_sub_ps(v2.m, v0.m),
								   _mm_sub_ps(v1.m, v0.m)), _mm_set1_ps(1.f)),
								   _mm_setzero_ps());
			return _mm_mul_ps(_mm_mul_ps(c, c),
							  _mm_sub_ps(_mm_set1_ps(3.0f), _mm_add_ps(c, c)));
		}

		friend inline const vec4 step(float f, const vec4 &v) {
			return _mm_and_ps(_mm_cmple_ps(v.m, _mm_set1_ps(f)),
							  _mm_set1_ps(1.0f));
		}

		friend inline const vec4 step(const vec4 &v0, const vec4 &v1) {
			return _mm_and_ps(_mm_cmple_ps(v0.m, v1.m), _mm_set1_ps(1.0f));
		}

		friend inline const vec4 trunc(const vec4 &v) {
			__m128 m = _mm_cmpunord_ps(v.m,
					   _mm_cmpge_ps(_mm_andnot_ps(_mm_set1_ps(-0.0f), v.m),
												  _mm_set1_ps(8388608.0f)));
			return _mm_or_ps(_mm_andnot_ps(m, _mm_cvtepi32_ps(
							 _mm_cvttps_epi32(v.m))), _mm_and_ps(m, v.m));
		}

		// ----------------------------------------------------------------- //

		friend inline float distance(const vec4 &v0, const vec4 &v1) {
			__m128 l = _mm_sub_ps(v0.m, v1.m);
			l = _mm_mul_ps(l, l);
			l = _mm_add_ps(l, _mm_shufd(l, 0x4E));
			return _mm_cvtss_f32(_mm_sqrt_ss(_mm_add_ss(l,
											 _mm_shufd(l, 0x11))));
		}

		friend inline float dot(const vec4 &v0, const vec4 &v1) {
			__m128 l = _mm_mul_ps(v0.m, v1.m);
			l = _mm_add_ps(l, _mm_shufd(l, 0x4E));
			return _mm_cvtss_f32(_mm_add_ss(l, _mm_shufd(l, 0x11)));
		}

		friend inline const vec4 faceforward(const vec4 &v0,
											 const vec4 &v1, const vec4 &v2) {
			__m128 l = _mm_mul_ps(v2.m, v1.m);
			l = _mm_add_ps(l, _mm_shufd(l, 0x4E));
			return _mm_xor_ps(_mm_and_ps(_mm_cmpnlt_ps(
					_mm_add_ps(l, _mm_shufd(l, 0x11)),
					_mm_setzero_ps()), _mm_set1_ps(-0.f)), v0.m);
		}

		friend inline float length(const vec4 &v) {
			__m128 l = _mm_mul_ps(v.m, v.m);
			l = _mm_add_ps(l, _mm_shufd(l, 0x4E));
			return _mm_cvtss_f32(_mm_sqrt_ss(_mm_add_ss(l,
											 _mm_shufd(l, 0x11))));
		}

		friend inline const vec4 normalize(const vec4 &v) {
			__m128 l = _mm_mul_ps(v.m, v.m);
			l = _mm_add_ps(l, _mm_shufd(l, 0x4E));
			return _mm_div_ps(v.m, _mm_sqrt_ps(_mm_add_ps(l,
											   _mm_shufd(l, 0x11))));
		}

		friend inline const vec4 reflect(const vec4 &v0, const vec4 &v1) {
			__m128 l = _mm_mul_ps(v1.m, v0.m);
			l = _mm_add_ps(l, _mm_shufd(l, 0x4E));
			l = _mm_add_ps(l, _mm_shufd(l, 0x11));
			return _mm_sub_ps(v0.m, _mm_mul_ps(_mm_add_ps(l, l), v1.m));
		}

		friend inline const vec4 refract(const vec4 &v0, const vec4 &v1,
										 float f) {
			__m128 o = _mm_set1_ps(1.0f);
			__m128 e = _mm_set1_ps(f);
			__m128 d = _mm_mul_ps(v1.m, v0.m);
			d = _mm_add_ps(d, _mm_shufd(d, 0x4E));
			d = _mm_add_ps(d, _mm_shufd(d, 0x11));
			__m128 k = _mm_sub_ps(o, _mm_mul_ps(_mm_mul_ps(e, e),
									 _mm_sub_ps(o, _mm_mul_ps(d, d))));
			return _mm_and_ps(_mm_cmpnlt_ps(k, _mm_setzero_ps()),
							  _mm_mul_ps(_mm_mul_ps(e, _mm_sub_ps(v0.m,
							  _mm_mul_ps(_mm_mul_ps(e, d), _mm_sqrt_ps(k)))),
										 v1.m));
		}

		// ----------------------------------------------------------------- //

		friend inline bool operator == (const vec4 &v0, const vec4 &v1) {
			return (_mm_movemask_ps(_mm_cmpeq_ps(v0.m, v1.m)) == 0xF);
		}

		friend inline bool operator != (const vec4 &v0, const vec4 &v1) {
			return (_mm_movemask_ps(_mm_cmpneq_ps(v0.m, v1.m)) != 0x0);
		}

		// ----------------------------------------------------------------- //

		union {
				// Vertex / Vector 
			struct {
				float x, y, z, w;
			};
				// Color
			struct {
				float r, g, b, a;
			};
				// Texture coordinates
			struct {
				float s, t, p, q;
			};

				// SSE register
			__m128	m;
		};

		// Avoid pollution
	#undef _mm_shufd
};

// Template specialization for mask 0xE4 (No shuffle)
template<>
inline __m128 vec4::shuffle<0xE4>(const __m128 &xmm) {
	return xmm;
}

#include "swizzle2.h"
#include "swizzle4.h"

#endif
