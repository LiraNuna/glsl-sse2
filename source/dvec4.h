#ifndef __dvec4_H__
#define __dvec4_H__

#include <emmintrin.h>

class dvec4
{
	public:
			// Empty constructor
		inline dvec4() {
			m1 = _mm_setzero_pd();
			m2 = _mm_setzero_pd();
		}

			// Fill constructor
		explicit inline dvec4(double d) {
			m1 = _mm_set1_pd(d);
			m2 = _mm_set1_pd(d);
		}

			// 4 var init constructor
		inline dvec4(double _x, double _y, double _z, double _w) {
			m1 = _mm_setr_pd(_x, _y);
			m2 = _mm_setr_pd(_z, _w);
		}

			// Integer array constructor
		inline dvec4(const double* dv) {
			m1 = _mm_loadu_pd(dv);
			m2 = _mm_loadu_pd(dv + 2);
		}

			// Copy constructor
		inline dvec4(const dvec4 &v) {
			m1 = v.m1;
			m2 = v.m2;
		}

			// SSE2 compatible constructor
		inline dvec4(const __m128d &_m1, const __m128d &_m2) {
			m1 = _m1;
			m2 = _m2;
		}

		// ----------------------------------------------------------------- //

			// Write direct access operator
		inline double& operator[](int index) {
			return ((double*)this)[index];
		}

			// Read direct access operator
		inline const double& operator[](int index) const {
			return ((const double*)this)[index];
		}

			// Cast operator
		inline operator double* () {
			return (double*)this;
		}

			// Const cast operator
		inline operator const double* () const {
			return (const double*)this;
		}

		// ----------------------------------------------------------------- //

		friend inline dvec4& operator += (dvec4 &v, double d) {
			__m128d dd = _mm_set1_pd(d);
			v.m1 = _mm_add_pd(v.m1, dd);
			v.m2 = _mm_add_pd(v.m2, dd);
			return v;
		}

		friend inline dvec4& operator += (dvec4 &v0, const dvec4 &v1) {
			v0.m1 = _mm_add_pd(v0.m1, v1.m1);
			v0.m2 = _mm_add_pd(v0.m2, v1.m2);
			return v0;
		}

		friend inline dvec4& operator -= (dvec4 &v, double d) {
			__m128d dd = _mm_set1_pd(d);
			v.m1 = _mm_sub_pd(v.m1, dd);
			v.m2 = _mm_sub_pd(v.m2, dd);
			return v;
		}

		friend inline dvec4& operator -= (dvec4 &v0, const dvec4 &v1) {
			v0.m1 = _mm_sub_pd(v0.m1, v1.m1);
			v0.m2 = _mm_sub_pd(v0.m2, v1.m2);
			return v0;
		}

		friend inline dvec4& operator *= (dvec4 &v, double d) {
			__m128d dd = _mm_set1_pd(d);
			v.m1 = _mm_mul_pd(v.m1, dd);
			v.m2 = _mm_mul_pd(v.m2, dd);
			return v;
		}

		friend inline dvec4& operator *= (dvec4 &v0, const dvec4 &v1) {
			v0.m1 = _mm_mul_pd(v0.m1, v1.m1);
			v0.m2 = _mm_mul_pd(v0.m2, v1.m2);
			return v0;
		}

		friend inline dvec4& operator /= (dvec4 &v, double d) {
			__m128d dd = _mm_set1_pd(d);
			v.m1 = _mm_div_pd(v.m1, dd);
			v.m2 = _mm_div_pd(v.m2, dd);
			return v;
		}

		friend inline dvec4& operator /= (dvec4 &v0, const dvec4 &v1) {
			v0.m1 = _mm_div_pd(v0.m1, v1.m1);
			v0.m2 = _mm_div_pd(v0.m2, v1.m2);
			return v0;
		}

		// ----------------------------------------------------------------- //

		friend inline const dvec4 operator + (double d, const dvec4 &v) {
			__m128d dd = _mm_set1_pd(d);
			return dvec4(_mm_add_pd(dd, v.m1), _mm_add_pd(dd, v.m2));
		}

		friend inline const dvec4 operator + (const dvec4 &v, double d) {
			__m128d dd = _mm_set1_pd(d);
			return dvec4(_mm_add_pd(v.m1, dd), _mm_add_pd(v.m2, dd));
		}

		friend inline const dvec4 operator + (const dvec4 &v0, const dvec4 &v1) {
			return dvec4(_mm_add_pd(v0.m1, v1.m1), _mm_add_pd(v0.m2, v1.m2));
		}

		friend inline const dvec4 operator - (const dvec4 &v) {
			__m128d nz = _mm_set1_pd(-0.0);
			return dvec4(_mm_xor_pd(v.m1, nz), _mm_xor_pd(v.m2, nz));
		}

		friend inline const dvec4 operator - (double d, const dvec4 &v) {
			__m128d dd = _mm_set1_pd(d);
			return dvec4(_mm_sub_pd(dd, v.m1), _mm_sub_pd(dd, v.m2));
		}

		friend inline const dvec4 operator - (const dvec4 &v, double d) {
			__m128d dd = _mm_set1_pd(d);
			return dvec4(_mm_sub_pd(v.m1, dd), _mm_sub_pd(v.m2, dd));
		}

		friend inline const dvec4 operator - (const dvec4 &v0, const dvec4 &v1) {
			return dvec4(_mm_sub_pd(v0.m1, v1.m1), _mm_sub_pd(v0.m2, v1.m2));
		}

		friend inline const dvec4 operator * (double d, const dvec4 &v) {
			__m128d dd = _mm_set1_pd(d);
			return dvec4(_mm_mul_pd(dd, v.m1), _mm_mul_pd(dd, v.m2));
		}

		friend inline const dvec4 operator * (const dvec4 &v, double d) {
			__m128d dd = _mm_set1_pd(d);
			return dvec4(_mm_mul_pd(v.m1, dd), _mm_mul_pd(v.m2, dd));
		}

		friend inline const dvec4 operator * (const dvec4 &v0, const dvec4 &v1) {
			return dvec4(_mm_mul_pd(v0.m1, v1.m1), _mm_mul_pd(v0.m2, v1.m2));
		}

		friend inline const dvec4 operator / (double d, const dvec4 &v) {
			__m128d dd = _mm_set1_pd(d);
			return dvec4(_mm_div_pd(dd, v.m1), _mm_div_pd(dd, v.m2));
		}

		friend inline const dvec4 operator / (const dvec4 &v, double d) {
			__m128d dd = _mm_set1_pd(d);
			return dvec4(_mm_div_pd(v.m1, dd), _mm_div_pd(v.m2, dd));
		}

		friend inline const dvec4 operator / (const dvec4 &v0, const dvec4 &v1) {
			return dvec4(_mm_div_pd(v0.m1, v1.m1), _mm_div_pd(v0.m2, v1.m2));
		}

		// ----------------------------------------------------------------- //

		friend inline const dvec4 sqrt(const dvec4 &v) {
			return dvec4(_mm_sqrt_pd(v.m1), _mm_sqrt_pd(v.m2));
		}

		friend inline const dvec4 inversesqrt(const dvec4 &v) {
			__m128d o = _mm_set1_pd(1.0);
			return dvec4(_mm_div_pd(o, _mm_sqrt_pd(v.m1)),
						 _mm_div_pd(o, _mm_sqrt_pd(v.m2)));
		}

		// ----------------------------------------------------------------- //

		friend inline const dvec4 abs(const dvec4 &v) {
			__m128d nz = _mm_set1_pd(-0.0);
			return dvec4(_mm_andnot_pd(nz, v.m1), _mm_andnot_pd(nz, v.m2));
		}

		friend inline const dvec4 ceil(const dvec4 &v) {
			__m128d o = _mm_set1_pd(1.0);
			__m128d nz = _mm_set1_pd(-0.0);
			__m128d hg = _mm_set1_pd(4.5036e+15);
			__m128d t1 = _mm_or_pd(_mm_and_pd(v.m1, nz), hg);
			__m128d t2 = _mm_or_pd(_mm_and_pd(v.m2, nz), hg);
			t1 = _mm_sub_pd(_mm_add_pd(v.m1, t1), t1);
			t2 = _mm_sub_pd(_mm_add_pd(v.m2, t2), t2);
			return dvec4(_mm_add_pd(t1, _mm_and_pd(_mm_cmpgt_pd(v.m1, t1), o)),
						 _mm_add_pd(t2, _mm_and_pd(_mm_cmpgt_pd(v.m1, t2), o)));
		}

		friend inline const dvec4 clamp(const dvec4 &v, double d1, double d2) {
			__m128d dd1 = _mm_set1_pd(d1);
			__m128d dd2 = _mm_set1_pd(d2);
			return dvec4(_mm_max_pd(_mm_min_pd(v.m1, dd2), dd1),
						 _mm_max_pd(_mm_min_pd(v.m2, dd2), dd1));
		}

		friend inline const dvec4 clamp(const dvec4 &v0,
										const dvec4 &v1, const dvec4 &v2) {
			return dvec4(_mm_max_pd(_mm_min_pd(v0.m1, v2.m1), v1.m1),
						 _mm_max_pd(_mm_min_pd(v0.m2, v2.m2), v1.m2));
		}

		friend inline const dvec4 floor(const dvec4 &v) {
			__m128d o = _mm_set1_pd(1.0);
			__m128d nz = _mm_set1_pd(-0.0);
			__m128d hg = _mm_set1_pd(4.5036e+15);
			__m128d t1 = _mm_or_pd(_mm_and_pd(v.m1, nz), hg);
			__m128d t2 = _mm_or_pd(_mm_and_pd(v.m2, nz), hg);
			t1 = _mm_sub_pd(_mm_add_pd(v.m1, t1), t1);
			t2 = _mm_sub_pd(_mm_add_pd(v.m2, t2), t2);
			return dvec4(_mm_sub_pd(t1, _mm_and_pd(_mm_cmplt_pd(v.m1, t1), o)),
						 _mm_sub_pd(t2, _mm_and_pd(_mm_cmplt_pd(v.m1, t2), o)));
		}

		friend inline const dvec4 fract(const dvec4 &v) {
			__m128d o = _mm_set1_pd(1.0);
			__m128d nz = _mm_set1_pd(-0.0);
			__m128d hg = _mm_set1_pd(4.5036e+15);
			__m128d t1 = _mm_or_pd(_mm_and_pd(v.m1, nz), hg);
			__m128d t2 = _mm_or_pd(_mm_and_pd(v.m2, nz), hg);
			t1 = _mm_sub_pd(_mm_add_pd(v.m1, t1), t1);
			t2 = _mm_sub_pd(_mm_add_pd(v.m2, t2), t2);
			return dvec4(_mm_sub_pd(v.m1, _mm_sub_pd(t1, _mm_and_pd(_mm_cmplt_pd(v.m1, t1), o))),
						 _mm_sub_pd(v.m2, _mm_sub_pd(t2, _mm_and_pd(_mm_cmplt_pd(v.m2, t2), o))));
		}

		friend inline const dvec4 max(const dvec4 &v, double d) {
			__m128d dd = _mm_set1_pd(d);
			return dvec4(_mm_max_pd(v.m1, dd), _mm_max_pd(v.m2, dd));
		}

		friend inline const dvec4 max(const dvec4 &v0, const dvec4 &v1) {
			return dvec4(_mm_max_pd(v0.m1, v1.m1), _mm_max_pd(v0.m2, v1.m2));
		}

		friend inline const dvec4 min(const dvec4 &v, double d) {
			__m128d dd = _mm_set1_pd(d);
			return dvec4(_mm_min_pd(v.m1, dd), _mm_min_pd(v.m2, dd));
		}

		friend inline const dvec4 min(const dvec4 &v0, const dvec4 &v1) {
			return dvec4(_mm_min_pd(v0.m1, v1.m1), _mm_min_pd(v0.m2, v1.m2));
		}

		friend inline const dvec4 mix(const dvec4 &v0, const dvec4 &v1,
									  double d) {
			__m128d o = _mm_set1_pd(1.0);
			__m128d dd = _mm_set1_pd(d);
			return dvec4(_mm_add_pd(_mm_mul_pd(v0.m1, _mm_sub_pd(o, dd)),
						 _mm_mul_pd(v1.m1, dd)),
						 _mm_add_pd(_mm_mul_pd(v0.m2, _mm_sub_pd(o, dd)),
						 _mm_mul_pd(v1.m2, dd)));
		}

		friend inline const dvec4 mix(const dvec4 &v0, const dvec4 &v1,
									  const dvec4 &v2) {
			__m128d o = _mm_set1_pd(1.0);
			return dvec4(_mm_add_pd(_mm_mul_pd(v0.m1, _mm_sub_pd(o, v2.m1)),
						 _mm_mul_pd(v1.m1, v2.m1)),
						 _mm_add_pd(_mm_mul_pd(v0.m2, _mm_sub_pd(o, v2.m2)),
						 _mm_mul_pd(v1.m2, v2.m2)));
		}

		friend inline const dvec4 mod(const dvec4 &v0, double d) {

		}

		friend inline const dvec4 mod(const dvec4 &v0, const dvec4 &v1) {

		}

		friend inline const dvec4 sign(const dvec4 &v) {
			__m128d o = _mm_set1_pd(1);
			__m128d z = _mm_setzero_pd();
			__m128d nz = _mm_set1_pd(-0.0);
			return dvec4(_mm_and_pd(_mm_or_pd(_mm_and_pd(v.m1, nz), o),
								    _mm_cmpneq_pd(v.m1, z)),
						 _mm_and_pd(_mm_or_pd(_mm_and_pd(v.m2, nz), o),
									_mm_cmpneq_pd(v.m2, z)));
		}

		friend inline const dvec4 smoothstep(double d1, double d2,
		                                     const dvec4 &v) {
			__m128d z = _mm_setzero_pd();
			__m128d o = _mm_set1_pd(1.0);
			__m128d t = _mm_set1_pd(3.0);
			__m128d dd1 = _mm_set1_pd(d1);
			__m128d dd2 = _mm_set1_pd(d2);
			__m128d r1 = _mm_max_pd(_mm_min_pd(_mm_div_pd(
								    _mm_sub_pd(v.m1, dd1),
									_mm_sub_pd(dd2, dd1)), o), z);
			__m128d r2 = _mm_max_pd(_mm_min_pd(_mm_div_pd(
									_mm_sub_pd(v.m2, dd1),
									_mm_sub_pd(dd2, dd1)), o), z);
			return dvec4(_mm_mul_pd(_mm_mul_pd(r1, r1),
									_mm_sub_pd(t, _mm_add_pd(r1, r1))),
						 _mm_mul_pd(_mm_mul_pd(r2, r2),
									_mm_sub_pd(t, _mm_add_pd(r2, r2))));
		}

		friend inline const dvec4 smoothstep(const dvec4 &v0,
		                                     const dvec4 &v1, const dvec4 &v2) {
			__m128d z = _mm_setzero_pd();
			__m128d o = _mm_set1_pd(1.0);
			__m128d t = _mm_set1_pd(3.0);
			__m128d r1 = _mm_max_pd(_mm_min_pd(_mm_div_pd(
								    _mm_sub_pd(v2.m1, v0.m1),
									_mm_sub_pd(v1.m1, v0.m1)), o), z);
			__m128d r2 = _mm_max_pd(_mm_min_pd(_mm_div_pd(
									_mm_sub_pd(v2.m2, v0.m2),
									_mm_sub_pd(v1.m2, v0.m2)), o), z);
			return dvec4(_mm_mul_pd(_mm_mul_pd(r1, r1),
									_mm_sub_pd(t, _mm_add_pd(r1, r1))),
						 _mm_mul_pd(_mm_mul_pd(r2, r2),
									_mm_sub_pd(t, _mm_add_pd(r2, r2))));
		}

		friend inline const dvec4 step(double d, const dvec4 &v) {
			__m128d o = _mm_set1_pd(1.0);
			__m128d dd = _mm_set1_pd(d);
			return dvec4(_mm_and_pd(_mm_cmple_pd(v.m1, dd), o),
						 _mm_and_pd(_mm_cmple_pd(v.m2, dd), o));
		}

		friend inline const dvec4 step(const dvec4 &v0, const dvec4 &v1) {
			__m128d o = _mm_set1_pd(1.0);
			return dvec4(_mm_and_pd(_mm_cmple_pd(v0.m1, v1.m1), o),
						 _mm_and_pd(_mm_cmple_pd(v0.m2, v1.m2), o));
		}
/*
		friend inline const dvec4 trunc(const dvec4 &v) {

		}
*/
		// ----------------------------------------------------------------- //

		friend inline bool operator == (const dvec4 &v0, const dvec4 &v1) {
			return _mm_movemask_ps((_mm_shuffle_ps(
				_mm_castpd_ps(_mm_cmpeq_pd(v0.m1, v1.m1)),
				_mm_castpd_ps(_mm_cmpeq_pd(v0.m2, v1.m2)), 0x88))) == 0xF;
		}

		friend inline bool operator != (const dvec4 &v0, const dvec4 &v1) {
			return _mm_movemask_ps(_mm_shuffle_ps(
				_mm_castpd_ps(_mm_cmpneq_pd(v0.m1, v1.m1)),
				_mm_castpd_ps(_mm_cmpneq_pd(v0.m2, v1.m2)), 0x88)) != 0x0;
		}

		// ----------------------------------------------------------------- //

		union {
				// Vertex / Vector
			struct {
				double x, y, z, w;
			};
				// Color
			struct {
				double r, g, b, a;
			};
				// Texture coordinates
			struct {
				double s, t, p, q;
			};

				// SSE2 registers
			struct {
				__m128d	m1;
				__m128d	m2;
			};
		};
};

#include "swizzle.h"

#endif
