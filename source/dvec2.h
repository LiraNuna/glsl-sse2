#ifndef __DVEC2_H__
#define __DVEC2_H__

#include <emmintrin.h>

#include "dvec4.h"

class dvec2
{
	private:
			// Merges mask `target` with `m` into one unified mask that does the same sequential shuffle
		template <unsigned target, unsigned m>
		struct _mask_merger
		{
			enum
			{
				ROW0 = ((target >> (((m >> 0) & 1) << 1)) & 1) << 0,
				ROW1 = ((target >> (((m >> 1) & 1) << 1)) & 1) << 1,

				MASK = ROW0 | ROW1,
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

				MASK = ROW0 | ROW1,
			};

			private:
				_mask_reverser();
		};

			// Splits a mask to two low and high masks
		template <unsigned mask>
		struct _mask_splitter
		{
			enum
			{
				HI = ((mask >> 0) & 1) | ((mask >> 2) & 1) << 1,
				LO = ((mask >> 4) & 1) | ((mask >> 6) & 1) << 1,
			};

			private:
				_mask_splitter();
		};

			// Swizzle helper (Read only)
		template <unsigned mask>
		struct _swzl_ro
		{
			friend class dvec2;

			public:
				inline operator const dvec2 () const {
					return _mm_shuffle_pd(v.m, v.m, mask);
				}

				inline double operator[](int index) const {
					return v[(mask >> (index << 1)) & 0x3];
				}

					// Swizzle of the swizzle, read only const (2)
				template<unsigned other_mask>
				inline _swzl_ro<_mask_merger<mask, other_mask>::MASK> shuffle2_ro2() const {
					typedef _mask_merger<mask, other_mask> merged;
					return _swzl_ro<merged::MASK>(v);
				}

					// Swizzle of the swizzle, read/write const
				template<unsigned other_mask>
				inline _swzl_ro<_mask_merger<mask, other_mask>::MASK> shuffle2_rw2() const {
					typedef _mask_merger<mask, other_mask> merged;
					return _swzl_ro<merged::MASK>(v);
				}

				const double &x, &y;
				const double &r, &g;
				const double &s, &t;

			private:
					// This massive constructor maps a vector to references
				inline _swzl_ro(const dvec2 &v):
					x(v[(mask >> 0) & 0x1]), y(v[(mask >> 1) & 0x1]),
					r(v[(mask >> 0) & 0x1]), g(v[(mask >> 1) & 0x1]),
					s(v[(mask >> 0) & 0x1]), t(v[(mask >> 1) & 0x1]),

					v(v) {
						// Empty
				}

					// Reference to unswizzled self
				const dvec2 &v;
		};

			// Swizzle helper (Read/Write)
		template <unsigned mask>
		struct _swzl_rw
		{
			friend class dvec2;

			public:
				inline operator const dvec2 () const {
					return _mm_shuffle_pd(v.m, v.m, mask);
				}

				inline double& operator[](int index) {
					return v[(mask >> (index << 1)) & 0x3];
				}

					// Swizzle from dvec2
				inline dvec2& operator = (const dvec2 &r) {
					return v = _mm_shuffle_pd(r.m, r.m, _mask_reverser<mask>::MASK);
				}

					// Swizzle from same r/o mask (v1.xyzw = v2.xyzw)
				inline dvec2& operator = (const _swzl_ro<mask> &s) {
					return v = s.v;
				}

					// Swizzle from same mask (v1.xyzw = v2.xyzw)
				inline dvec2& operator = (const _swzl_rw &s) {
					return v = s.v;
				}

					// Swizzle mask => other_mask, r/o (v1.zwxy = v2.xyxy)
				template<unsigned other_mask>
				inline dvec2& operator = (const _swzl_ro<other_mask> &s) {
					typedef _mask_merger<other_mask, _mask_reverser<mask>::MASK> merged;

					return v = _mm_shuffle_pd(s.v.m, s.v.m, merged::MASK);
				}

					// Swizzle mask => other_mask (v1.zwxy = v2.xyzw)
				template<unsigned other_mask>
				inline dvec2& operator = (const _swzl_rw<other_mask> &s) {
					typedef _mask_merger<other_mask, _mask_reverser<mask>::MASK> merged;

					return v = _mm_shuffle_pd(s.v.m, s.v.m, merged::MASK);
				}

					// Swizzle of the swizzle, read only (v.xxxx.yyyy) (2)
				template<unsigned other_mask>
				inline _swzl_ro<_mask_merger<mask, other_mask>::MASK> shuffle4_ro2() const {
					typedef _mask_merger<mask, other_mask> merged;

					return _swzl_ro<merged::MASK>(v);
				}

					// Swizzle of the swizzle, read only (v.xxxx.yyyy) (4)
				template<unsigned other_mask>
				inline _swzl_ro<_mask_merger<mask, other_mask>::MASK> shuffle2_ro2() const {
					typedef _mask_merger<mask, other_mask> merged;

					return _swzl_ro<merged::MASK>(v);
				}

					// Swizzle of the swizzle, read/write (v1.zyxw.wzyx = ...)
				template<unsigned other_mask>
				inline _swzl_rw<_mask_merger<mask, other_mask>::MASK> shuffle2_rw2() {
					typedef _mask_merger<mask, other_mask> merged;

					return _swzl_rw<merged::MASK>(v);
				}

				// ----------------------------------------------------------------- //

				inline dvec2& operator += (double s) {
					return v += s;
				}

				inline dvec2& operator += (const dvec2 &v0) {
					return v += v0.shuffle2_ro2<mask>();
				}

				inline dvec2& operator -= (double s) {
					return v -= s;
				}

				inline dvec2& operator -= (const dvec2 &v0) {
					return v -= v0.shuffle2_ro2<mask>();
				}

				inline dvec2& operator *= (double s) {
					return v *= s;
				}

				inline dvec2& operator *= (const dvec2 &v0) {
					return v *= v0.shuffle2_ro2<mask>();
				}

				inline dvec2& operator /= (double s) {
					return v /= s;
				}

				inline dvec2& operator /= (const dvec2 &v0) {
					return v /= v0.shuffle2_ro2<mask>();
				}

				// ----------------------------------------------------------------- //

				double &x, &y;
				double &r, &g;
				double &s, &t;

			private:
					// This massive contructor maps a vector to references
				inline _swzl_rw(dvec2 &v):
					x(v[(mask >> 0) & 0x1]), y(v[(mask >> 1) & 0x1]),
					r(v[(mask >> 0) & 0x1]), g(v[(mask >> 1) & 0x1]),
					s(v[(mask >> 0) & 0x1]), t(v[(mask >> 1) & 0x1]),

					v(v) {
						// Empty
				}

					// Reference to unswizzled self
				dvec2 &v;
		};

	public:
			// Empty constructor
		inline dvec2() {
			m = _mm_setzero_pd();
		}

			// Fill constructor
		explicit inline dvec2(double d) {
			m = _mm_set1_pd(d);
		}

			// 4 var init constructor
		inline dvec2(double _x, double _y) {
			m = _mm_setr_pd(_x, _y);
		}

			// Integer array constructor
		inline dvec2(const double* dv) {
			m = _mm_loadu_pd(dv);
		}

			// Copy constructor
		inline dvec2(const dvec2 &v) {
			m = v.m;
		}

			// SSE2 compatible constructor
		inline dvec2(const __m128d &_m) {
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

			// Write direct access operator
		inline double& operator[](int index) {
			return reinterpret_cast<double *>(this)[index];
		}

			// Read direct access operator
		inline const double& operator[](int index) const {
			return reinterpret_cast<const double*>(this)[index];
		}

			// Cast operator
		inline operator double* () {
			return reinterpret_cast<double *>(this);
		}

			// Const cast operator
		inline operator const double* () const {
			return reinterpret_cast<const double *>(this);
		}

		// ----------------------------------------------------------------- //

			// Read-write swizzle
		template<unsigned mask>
		inline _swzl_rw<mask> shuffle2_rw2() {
			return _swzl_rw<mask>(*this);
		}

			// Read-write swizzle, const, actually read only
		template<unsigned mask>
		inline _swzl_ro<mask> shuffle2_rw2() const {
			return _swzl_ro<mask>(*this);
		}

			// Read-only swizzle (2)
		template<unsigned mask>
		inline _swzl_ro<mask> shuffle2_ro2() const {
			return _swzl_ro<mask>(*this);
		}

			// Read-only swizzle (4)
		template<unsigned mask>
		inline dvec4 shuffle4_ro2() const {
			return dvec4(_mm_shuffle_pd(m, m, _mask_splitter<mask>::HI),
						 _mm_shuffle_pd(m, m, _mask_splitter<mask>::LO));
		}

		// ----------------------------------------------------------------- //

		friend inline dvec2& operator += (dvec2 &v, double d) {
			v.m = _mm_add_pd(v.m, _mm_set1_pd(d));
			return v;
		}

		friend inline dvec2& operator += (dvec2 &v0, const dvec2 &v1) {
			v0.m = _mm_add_pd(v0.m, v1.m);
			return v0;
		}

		friend inline dvec2& operator -= (dvec2 &v, double d) {
			v.m = _mm_sub_pd(v.m, _mm_set1_pd(d));
			return v;
		}

		friend inline dvec2& operator -= (dvec2 &v0, const dvec2 &v1) {
			v0.m = _mm_sub_pd(v0.m, v1.m);
			return v0;
		}

		friend inline dvec2& operator *= (dvec2 &v, double d) {
			v.m = _mm_mul_pd(v.m, _mm_set1_pd(d));
			return v;
		}

		friend inline dvec2& operator *= (dvec2 &v0, const dvec2 &v1) {
			v0.m = _mm_mul_pd(v0.m, v1.m);
			return v0;
		}

		friend inline dvec2& operator /= (dvec2 &v, double d) {
			v.m = _mm_div_pd(v.m, _mm_set1_pd(d));
			return v;
		}

		friend inline dvec2& operator /= (dvec2 &v0, const dvec2 &v1) {
			v0.m = _mm_div_pd(v0.m, v1.m);
			return v0;
		}

		// ----------------------------------------------------------------- //

		friend inline const dvec2 operator + (double d, const dvec2 &v) {
			return _mm_add_pd(_mm_set1_pd(d), v.m);
		}

		friend inline const dvec2 operator + (const dvec2 &v, double d) {
			return _mm_add_pd(v.m, _mm_set1_pd(d));
		}

		friend inline const dvec2 operator + (const dvec2 &v0, const dvec2 &v1) {
			return _mm_add_pd(v0.m, v1.m);
		}

		friend inline const dvec2 operator - (const dvec2 &v) {
			return _mm_xor_pd(v.m, _mm_set1_pd(-0.0));
		}

		friend inline const dvec2 operator - (double d, const dvec2 &v) {
			return _mm_sub_pd(_mm_set1_pd(d), v.m);
		}

		friend inline const dvec2 operator - (const dvec2 &v, double d) {
			return _mm_sub_pd(v.m, _mm_set1_pd(d));
		}

		friend inline const dvec2 operator - (const dvec2 &v0, const dvec2 &v1) {
			return _mm_sub_pd(v0.m, v1.m);
		}

		friend inline const dvec2 operator * (double d, const dvec2 &v) {
			return _mm_mul_pd(_mm_set1_pd(d), v.m);
		}

		friend inline const dvec2 operator * (const dvec2 &v, double d) {
			return _mm_mul_pd(v.m, _mm_set1_pd(d));
		}

		friend inline const dvec2 operator * (const dvec2 &v0, const dvec2 &v1) {
			return _mm_mul_pd(v0.m, v1.m);
		}

		friend inline const dvec2 operator / (double d, const dvec2 &v) {
			return _mm_div_pd(_mm_set1_pd(d), v.m);
		}

		friend inline const dvec2 operator / (const dvec2 &v, double d) {
			return _mm_div_pd(v.m, _mm_set1_pd(d));
		}

		friend inline const dvec2 operator / (const dvec2 &v0, const dvec2 &v1) {
			return _mm_div_pd(v0.m, v1.m);
		}

		// ----------------------------------------------------------------- //
/*
		friend inline const dvec2 pow(const dvec2 &v0, const dvec2 &v1) {
			// TODO
		}

		friend inline const dvec2 exp(const dvec2 &v) {
			// TODO
		}
*/
		friend inline const dvec2 log(const dvec2 &v) {
			return log2(v) * 0.69314718055994530942;
		}
/*
		friend inline const dvec2 exp2(const dvec2 &v) {
			// TODO
		}
*/
		friend inline const dvec2 log2(const dvec2 &v) {
			__m128d o = _mm_set1_pd(1.0);
			__m128d c = _mm_castsi128_pd(_mm_set1_epi64x(0x7FF0000000000000LL));
			__m128d f = _mm_sub_pd(_mm_or_pd(_mm_andnot_pd(c, v.m),
											 _mm_and_pd(c, o)), o);
			__m128i a = _mm_sub_epi32(_mm_srli_epi32(_mm_castpd_si128(v.m), 20),
									  _mm_set1_epi32(1023));
			__m128d hi = _mm_add_pd(_mm_mul_pd(_mm_set1_pd( 3.61276447184348752E-05), f),
									_mm_set1_pd(-4.16662127033480827E-04));
			__m128d lo = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(-1.43988260692073185E-01), f),
									_mm_set1_pd( 1.60245637034704267E-01));
			hi = _mm_add_pd(_mm_mul_pd(f, hi), _mm_set1_pd( 2.28193656337578229E-03));
			lo = _mm_add_pd(_mm_mul_pd(f, lo), _mm_set1_pd(-1.80329036970820794E-01));
			hi = _mm_add_pd(_mm_mul_pd(f, hi), _mm_set1_pd(-7.93793829370930689E-03));
			lo = _mm_add_pd(_mm_mul_pd(f, lo), _mm_set1_pd( 2.06098446037376922E-01));
			hi = _mm_add_pd(_mm_mul_pd(f, hi), _mm_set1_pd( 1.98461565426430164E-02));
			lo = _mm_add_pd(_mm_mul_pd(f, lo), _mm_set1_pd(-2.40449108727688962E-01));
			hi = _mm_add_pd(_mm_mul_pd(f, hi), _mm_set1_pd(-3.84093543662501949E-02));
			lo = _mm_add_pd(_mm_mul_pd(f, lo), _mm_set1_pd( 2.88539004851839364E-01));
			hi = _mm_add_pd(_mm_mul_pd(f, hi), _mm_set1_pd( 6.08335872067172597E-02));
			lo = _mm_add_pd(_mm_mul_pd(f, lo), _mm_set1_pd(-3.60673760117245982E-01));
			hi = _mm_add_pd(_mm_mul_pd(f, hi), _mm_set1_pd(-8.27937055456904317E-02));
			lo = _mm_add_pd(_mm_mul_pd(f, lo), _mm_set1_pd( 4.80898346961226595E-01));
			hi = _mm_add_pd(_mm_mul_pd(f, hi), _mm_set1_pd( 1.01392360727236079E-01));
			lo = _mm_add_pd(_mm_mul_pd(f, lo), _mm_set1_pd(-7.21347520444469934E-01));
			hi = _mm_add_pd(_mm_mul_pd(f, hi), _mm_set1_pd(-1.16530490533844182E-01));
			lo = _mm_add_pd(_mm_mul_pd(f, lo), _mm_set1_pd( 0.44269504088896339E+00));
			hi = _mm_add_pd(_mm_mul_pd(f, hi), _mm_set1_pd( 1.30009193360025350E-01));
			__m128d x2  = _mm_mul_pd(f, f);
			__m128d x10 = _mm_mul_pd(x2, x2);
			x10 = _mm_mul_pd(x10, x10);
			x10 = _mm_mul_pd(x2, x10);
			return _mm_add_pd(_mm_add_pd(_mm_mul_pd(
							  _mm_add_pd(_mm_mul_pd(x10, hi), lo), f), f),
										 _mm_cvtepi32_pd(_mm_shuffle_epi32(a, 0x0D)));
		}

		friend inline const dvec2 sqrt(const dvec2 &v) {
			return _mm_sqrt_pd(v.m);
		}

		friend inline const dvec2 inversesqrt(const dvec2 &v) {
			return _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(v.m));
		}

		// ----------------------------------------------------------------- //

		friend inline const dvec2 abs(const dvec2 &v) {
			return _mm_andnot_pd(_mm_set1_pd(-0.0), v.m);
		}

		friend inline const dvec2 ceil(const dvec2 &v) {
			return _mm_cvtepi32_pd(_mm_cvtpd_epi32(_mm_add_pd(v.m, _mm_set1_pd(0.5))));
		}

		friend inline const dvec2 clamp(const dvec2 &v, double d1, double d2) {
			return _mm_max_pd(_mm_min_pd(v.m, _mm_set1_pd(d2)),
											  _mm_set1_pd(d1));
		}

		friend inline const dvec2 clamp(const dvec2 &v0,
										const dvec2 &v1, const dvec2 &v2) {
			return _mm_max_pd(_mm_min_pd(v0.m, v2.m), v1.m);
		}

		friend inline const dvec2 floor(const dvec2 &v) {
			return _mm_cvtepi32_pd(_mm_srai_epi32(_mm_cvtpd_epi32(
								   _mm_sub_pd(_mm_add_pd(v.m, v.m), _mm_set1_pd(0.5))), 1));
		}

		friend inline const dvec2 fract(const dvec2 &v) {
			return _mm_sub_pd(v.m, _mm_cvtepi32_pd(_mm_srai_epi32(
								   _mm_cvtpd_epi32(_mm_sub_pd(_mm_add_pd(v.m, v.m),
												   _mm_set1_pd(0.5))), 1)));
		}

		friend inline const dvec2 max(const dvec2 &v, double d) {
			return _mm_max_pd(v.m, _mm_set1_pd(d));
		}

		friend inline const dvec2 max(const dvec2 &v0, const dvec2 &v1) {
			return _mm_max_pd(v0.m, v1.m);
		}

		friend inline const dvec2 min(const dvec2 &v, double d) {
			return _mm_min_pd(v.m, _mm_set1_pd(d));
		}

		friend inline const dvec2 min(const dvec2 &v0, const dvec2 &v1) {
			return _mm_min_pd(v0.m, v1.m);
		}

		friend inline const dvec2 mix(const dvec2 &v0, const dvec2 &v1,
									  double d) {
			__m128d dd = _mm_set1_pd(d);
			return _mm_add_pd(_mm_mul_pd(v0.m, _mm_sub_pd(_mm_set1_pd(1.0), dd)),
							  _mm_mul_pd(v1.m, dd));
		}

		friend inline const dvec2 mix(const dvec2 &v0, const dvec2 &v1,
									  const dvec2 &v2) {
			return _mm_add_pd(_mm_mul_pd(v0.m, _mm_sub_pd(_mm_set1_pd(1.0), v2.m)),
							  _mm_mul_pd(v1.m, v2.m));
		}

		friend inline const dvec2 mod(const dvec2 &v, double d) {
			__m128d dd = _mm_set1_pd(d);
			__m128d d1 = _mm_div_pd(v.m, dd);
			return _mm_sub_pd(v.m, _mm_mul_pd(dd, _mm_cvtepi32_pd(
								   _mm_srai_epi32(_mm_cvtpd_epi32(_mm_sub_pd(
								   _mm_add_pd(d1, d1), _mm_set1_pd(0.5))), 1))));
		}

		friend inline const dvec2 mod(const dvec2 &v0, const dvec2 &v1) {
			__m128d d1 = _mm_div_pd(v0.m, v1.m);
			return _mm_sub_pd(v0.m, _mm_mul_pd(v1.m, _mm_cvtepi32_pd(
									_mm_srai_epi32(_mm_cvtpd_epi32(_mm_sub_pd(
									_mm_add_pd(d1, d1), _mm_set1_pd(0.5))), 1))));
		}

		friend inline const dvec2 modf(const dvec2 &v0, dvec2 &v1) {
			v1.m = _mm_or_pd(_mm_cvtepi32_pd(_mm_cvttpd_epi32(v0.m)),
							 _mm_and_pd(_mm_set1_pd(-0.0), v0.m));
			return _mm_sub_pd(v0.m, v1.m);
		}

		friend inline const dvec2 sign(const dvec2 &v) {
			return _mm_and_pd(_mm_or_pd(_mm_and_pd(v.m, _mm_set1_pd(-0.0)), _mm_set1_pd(1)),
							  _mm_cmpneq_pd(v.m, _mm_setzero_pd()));
		}

		friend inline const dvec2 smoothstep(double d1, double d2,
											 const dvec2 &v) {
			__m128d dd1 = _mm_set1_pd(d1);
			__m128d c = _mm_max_pd(_mm_min_pd(_mm_div_pd(_mm_sub_pd(v.m, dd1),
								   _mm_sub_pd(_mm_set1_pd(d2), dd1)),
								   _mm_set1_pd(1.0)), _mm_setzero_pd());
			return _mm_mul_pd(_mm_mul_pd(c, c),
							  _mm_sub_pd(_mm_set1_pd(3.0), _mm_add_pd(c, c)));
		}

		friend inline const dvec2 smoothstep(const dvec2 &v0,
											 const dvec2 &v1, const dvec2 &v2) {
			 __m128d c = _mm_max_pd(_mm_min_pd(_mm_div_pd(_mm_sub_pd(v2.m, v0.m),
								    _mm_sub_pd(v1.m, v0.m)), _mm_set1_pd(1.0)),
								    _mm_setzero_pd());
			return _mm_mul_pd(_mm_mul_pd(c, c),
							  _mm_sub_pd(_mm_set1_pd(3.0), _mm_add_pd(c, c)));
		}

		friend inline const dvec2 step(double d, const dvec2 &v) {
			return _mm_and_pd(_mm_cmple_pd(v.m, _mm_set1_pd(d)),
												_mm_set1_pd(1.0));
		}

		friend inline const dvec2 step(const dvec2 &v0, const dvec2 &v1) {
			return _mm_and_pd(_mm_cmple_pd(v0.m, v1.m), _mm_set1_pd(1.0));
		}

		friend inline const dvec2 trunc(const dvec2 &v) {
			return _mm_cvtepi32_pd(_mm_cvtpd_epi32(_mm_sub_pd(v.m,
								   _mm_or_pd(_mm_and_pd(v.m, _mm_set1_pd(-0.0)),
														     _mm_set1_pd(0.5)))));
		}

		// ----------------------------------------------------------------- //

		friend inline double distance(const dvec2 &v0, const dvec2 &v1) {
			__m128d d = _mm_sub_pd(v0.m, v1.m);
			__m128d l = _mm_mul_pd(d, d);
			return _mm_cvtsd_f64(_mm_add_pd(l, _mm_shuffle_pd(l, l, 0x01)));
		}

		friend inline double dot(const dvec2 &v0, const dvec2 &v1) {
			__m128d l = _mm_mul_pd(v0.m, v1.m);
			return _mm_cvtsd_f64(_mm_add_pd(l, _mm_shuffle_pd(l, l, 0x01)));
		}

		friend inline const dvec2 faceforward(const dvec2 &v0,
											  const dvec2 &v1, const dvec2 &v2) {
			__m128d l = _mm_mul_pd(v2.m, v1.m);
			return _mm_xor_pd(_mm_and_pd(_mm_cmpnlt_pd(
					_mm_add_pd(l, _mm_shuffle_pd(l, l, 0x01)),
					_mm_setzero_pd()), _mm_set1_pd(-0.f)), v0.m);
		}

		friend inline double length(const dvec2 &v) {
			__m128d l = _mm_mul_pd(v.m, v.m);
			return _mm_cvtsd_f64(_mm_sqrt_pd(_mm_add_pd(l, _mm_shuffle_pd(l, l, 0x01))));
		}

		friend inline const dvec2 normalize(const dvec2 &v) {
			__m128d l = _mm_mul_pd(v.m, v.m);
			return _mm_div_pd(v.m, _mm_sqrt_pd(_mm_add_pd(l, _mm_shuffle_pd(l, l, 0x01))));
		}

		friend inline const dvec2 reflect(const dvec2 &v0, const dvec2 &v1) {
			__m128d l = _mm_mul_pd(v0.m, v1.m);
			__m128d d = _mm_add_pd(l, _mm_shuffle_pd(l, l, 0x01));
			return _mm_sub_pd(v0.m, _mm_mul_pd(_mm_add_pd(d, d), v1.m));

		}

		friend inline const dvec2 refract(const dvec2 &v0, const dvec2 &v1,
										  double d) {
			__m128d o = _mm_set1_pd(1.0);
			__m128d e = _mm_set1_pd(d);
			__m128d l = _mm_mul_pd(v0.m, v1.m);
			__m128d dd = _mm_add_pd(l, _mm_shuffle_pd(l, l, 0x01));
			__m128d k = _mm_sub_pd(o, _mm_mul_pd(_mm_mul_pd(e, e),
									  _mm_sub_pd(o, _mm_mul_pd(dd, dd))));
			return _mm_and_pd(_mm_cmpnlt_pd(k, _mm_setzero_pd()),
							  _mm_mul_pd(_mm_mul_pd(e, _mm_sub_pd(v0.m,
							  _mm_mul_pd(_mm_mul_pd(e, dd), _mm_sqrt_pd(k)))), v1.m));
		}

		// ----------------------------------------------------------------- //

		friend inline bool operator == (const dvec2 &v0, const dvec2 &v1) {
			return _mm_movemask_pd(_mm_cmpeq_pd(v0.m, v1.m)) == 0x03;
		}

		friend inline bool operator != (const dvec2 &v0, const dvec2 &v1) {
			return _mm_movemask_pd(_mm_cmpneq_pd(v0.m, v1.m)) != 0x00;
		}

		// ----------------------------------------------------------------- //

		union {
				// Vertex / Vector
			struct {
				double x, y;
			};
				// Color
			struct {
				double r, g;
			};
				// Texture coordinates
			struct {
				double s, t;
			};

				// SSE2 registers
			__m128d m;
		};
};

#include "swizzle2.h"
#include "swizzle4.h"

#endif
