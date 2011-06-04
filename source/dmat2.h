#include "dvec2.h"

#ifndef __DMAT2_H__
#define __DMAT2_H__

class dmat2
{
	public:
			// Identity matrix
		inline dmat2() {
			m1 = _mm_setr_pd(1.0, 0.0);
			m2 = _mm_setr_pd(0.0, 0.0);
		}

			// Scaled matrix
		explicit inline dmat2(double d) {
			m1 = _mm_setr_pd(  d, 0.0);
			m2 = _mm_setr_pd(0.0,   d);
		}

			// 4 vectors constructor
		inline dmat2(const dvec2 &_v1, const dvec2 &_v2) {
			m1 = _v1.m;
			m2 = _v2.m;
		}

			// Full scalar constructor
		inline dmat2(double  _d1, double  _d2, double  _d3, double  _d4) {
			m1 = _mm_setr_pd( _d1,  _d2);
			m2 = _mm_setr_pd( _d3,  _d4);
		}

			// Copy constructor
		inline dmat2(const dmat2 &m) {
			m1 = m.m1;
			m2 = m.m2;
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
		inline dvec2& operator[](int index) {
			return reinterpret_cast<dvec2 &>(m[index]);
		}

			// Read direct access operator
		inline const dvec2& operator[](int index) const {
			return reinterpret_cast<const dvec2 &>(m[index]);
		}

			// Cast operator
		inline operator double*() {
			return reinterpret_cast<double *>(this);
		}

			// Const cast operator
		inline operator const double*() const {
			return reinterpret_cast<const double *>(this);
		}

		// ----------------------------------------------------------------- //

		inline dmat2& operator += (double d) {
			__m128d dd = _mm_set1_pd(d);
			m1 = _mm_add_pd(m1, dd);
			m2 = _mm_add_pd(m2, dd);

			return *this;
		}

		inline dmat2& operator += (const dmat2 &m) {
			m1 = _mm_add_pd(m1, m.m1);
			m2 = _mm_add_pd(m2, m.m2);

			return *this;
		}

		inline dmat2& operator -= (double d) {
			__m128d dd = _mm_set1_pd(d);
			m1 = _mm_sub_pd(m1, dd);
			m2 = _mm_sub_pd(m2, dd);

			return *this;
		}

		inline dmat2& operator -= (const dmat2 &m) {
			m1 = _mm_sub_pd(m1, m.m1);
			m2 = _mm_sub_pd(m2, m.m2);

			return *this;
		}

		inline dmat2& operator *= (double d) {
			__m128d dd = _mm_set1_pd(d);
			m1 = _mm_mul_pd(m1, dd);
			m2 = _mm_mul_pd(m2, dd);

			return *this;
		}

		inline dmat2& operator *= (const dmat2 &m) {
			m1 = _mm_add_pd(_mm_mul_pd(_mm_unpacklo_pd(m1, m1), m.m1),
							_mm_mul_pd(_mm_unpackhi_pd(m1, m1), m.m2));
			m2 = _mm_add_pd(_mm_mul_pd(_mm_unpacklo_pd(m2, m2), m.m1),
							_mm_mul_pd(_mm_unpackhi_pd(m2, m2), m.m2));

			return *this;
		}

		inline dmat2& operator /= (double d) {
			__m128d dd = _mm_set1_pd(d);
			m1 = _mm_div_pd(m1, dd);
			m2 = _mm_div_pd(m2, dd);

			return *this;
		}

		inline dmat2& operator /= (const dmat2 &m) {
			m1 = _mm_div_pd(m1, m.m1);
			m2 = _mm_div_pd(m2, m.m2);

			return *this;
		}

		// ----------------------------------------------------------------- //

		friend inline dmat2 operator + (const dmat2 &m, double d) {
			__m128d dd = _mm_set1_pd(d);
			return dmat2(_mm_add_pd(m.m1, dd), _mm_add_pd(m.m2, dd));
		}

		friend inline dmat2 operator + (const dmat2 &m0, const dmat2 &m1) {
			return dmat2(_mm_add_pd(m0.m1, m1.m1), _mm_add_pd(m0.m2, m1.m2));
		}

		friend inline dmat2 operator - (const dmat2 &m, double d) {
			__m128d dd = _mm_set1_pd(d);
			return dmat2(_mm_sub_pd(m.m1, dd), _mm_sub_pd(m.m2, dd));
		}

		friend inline dmat2 operator - (double d, const dmat2 &m) {
			__m128d dd = _mm_set1_pd(d);
			return dmat2(_mm_sub_pd(dd, m.m1), _mm_sub_pd(dd, m.m2));
		}

		friend inline dmat2 operator - (const dmat2 &m0, const dmat2 &m1) {
			return dmat2(_mm_sub_pd(m0.m1, m1.m1), _mm_sub_pd(m0.m2, m1.m2));
		}

		friend inline dmat2 operator * (const dmat2 &m, double d) {
			__m128d dd = _mm_set1_pd(d);
			return dmat2(_mm_mul_pd(m.m1, dd), _mm_mul_pd(m.m2, dd));
		}

		friend inline dvec2 operator * (const dmat2 &m, const dvec2 &v) {
			return _mm_add_pd(_mm_mul_pd(_mm_unpacklo_pd(v.m, v.m), m.m1),
							  _mm_mul_pd(_mm_unpackhi_pd(v.m, v.m), m.m2));
		}

		friend inline dvec2 operator * (const dvec2 &v, const dmat2 &m) {
			return _mm_add_pd(_mm_mul_pd(_mm_unpacklo_pd(v.m, v.m),
										 _mm_unpacklo_pd(m.m1, m.m2)),
							  _mm_mul_pd(_mm_unpackhi_pd(v.m, v.m),
										 _mm_unpackhi_pd(m.m1, m.m2)));
		}

		friend inline dmat2 operator * (const dmat2 &m0, const dmat2 &m1) {
			return dmat2(_mm_add_pd(_mm_mul_pd(_mm_unpacklo_pd(m0.m1, m0.m1), m1.m1),
									_mm_mul_pd(_mm_unpackhi_pd(m0.m1, m0.m1), m1.m2)),
						 _mm_add_pd(_mm_mul_pd(_mm_unpacklo_pd(m0.m2, m0.m2), m1.m1),
									_mm_mul_pd(_mm_unpackhi_pd(m0.m2, m0.m2), m1.m2)));
		}

		friend inline dmat2 operator / (const dmat2 &m, double d) {
			__m128d dd = _mm_set1_pd(d);
			return dmat2(_mm_div_pd(m.m1, dd), _mm_div_pd(m.m2, dd));
		}

		friend inline dmat2 operator / (double d, const dmat2 &m) {
			__m128d dd = _mm_set1_pd(d);
			return dmat2(_mm_div_pd(dd, m.m1), _mm_div_pd(dd, m.m2));
		}

		friend inline dmat2 operator / (const dmat2 &m0, const dmat2 &m1) {
			return dmat2(_mm_div_pd(m0.m1, m1.m1), _mm_div_pd(m0.m2, m1.m2));
		}

		// ----------------------------------------------------------------- //

		friend inline dmat2 matrixCompMult(const dmat2 &m0, const dmat2 &m1) {
			return dmat2(_mm_mul_pd(m0.m1, m1.m1), _mm_mul_pd(m0.m2, m1.m2));
		}

		// ----------------------------------------------------------------- //

		friend inline dmat2 transpose(const dmat2 &m) {
			return dmat2(_mm_unpackhi_pd(m.m1, m.m2),
						 _mm_unpacklo_pd(m.m1, m.m2));
		}

		friend inline double determinant(const dmat2 &m) {
			__m128d d = _mm_mul_pd(m.m1, _mm_shuffle_pd(m.m2, m.m2, 0x01));
			return _mm_cvtsd_f64(_mm_sub_pd(d, _mm_shuffle_pd(d, d, 0x01)));
		}

		friend inline dmat2 inverse(const dmat2 &m) {
			__m128d d = _mm_mul_pd(m.m1, _mm_shuffle_pd(m.m2, m.m2, 0x01));
			d = _mm_sub_pd(d, _mm_shuffle_pd(d, d, 0x01));
			d = _mm_div_pd(_mm_set1_pd(1.0), _mm_unpacklo_pd(d, d));
			return dmat2(_mm_mul_pd(_mm_xor_pd(_mm_unpackhi_pd(m.m2, m.m1),
									_mm_set_pd(-0.0,  0.0)), d),
						 _mm_mul_pd(_mm_xor_pd(_mm_unpacklo_pd(m.m2, m.m1),
									_mm_set_pd( 0.0, -0.0)), d));
		}

		// ----------------------------------------------------------------- //

	private:
			// SSE constructor
		inline dmat2(const __m128d &_m1, const __m128d &_m2) {
			m1 = _m1;
			m2 = _m2;
		}

		union {
			__m128d m[2];
			struct {
				__m128d m1;
				__m128d m2;
			};

/*			// This code is waiting for unrestricted unions feature in c++0x
			dvec2 v[2];
			struct {
				dvec2 v1;
				dvec2 v2;
			};
*/
		};
};

#endif
