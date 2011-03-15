#ifndef __DVEC2_H__
#define __DVEC2_H__

#include <emmintrin.h>

class dvec2
{
	private:
		// TODO: Swizzle proxy class

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

		// TODO: Swizzle proxy providers

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

#include "swizzle4.h"

#endif
