#include "dvec4.h"

#ifndef __DMAT4_H__
#define __DMAT4_H__

class dmat4
{
	public:
			// Identity matrix
		inline dmat4() {
			m11 = _mm_setr_pd(1.0, 0.0);
			m12 = _mm_setr_pd(0.0, 0.0);

			m21 = _mm_setr_pd(0.0, 1.0);
			m22 = _mm_setr_pd(0.0, 0.0);

			m31 = _mm_setr_pd(0.0, 0.0);
			m32 = _mm_setr_pd(1.0, 0.0);

			m41 = _mm_setr_pd(0.0, 0.0);
			m42 = _mm_setr_pd(0.0, 1.0);
		}

			// Scaled matrix
		explicit inline dmat4(double d) {
			m11 = _mm_setr_pd(  d, 0.0);
			m12 = _mm_setr_pd(0.0, 0.0);

			m21 = _mm_setr_pd(0.0,   d);
			m22 = _mm_setr_pd(0.0, 0.0);

			m31 = _mm_setr_pd(0.0, 0.0);
			m32 = _mm_setr_pd(  d, 0.0);

			m41 = _mm_setr_pd(0.0, 0.0);
			m42 = _mm_setr_pd(0.0,   d);
		}

			// 4 vectors constructor
		inline dmat4(const dvec4 &_v1, const dvec4 &_v2,
		             const dvec4 &_v3, const dvec4 &_v4) {
			m11 = _v1.m1;
			m12 = _v1.m2;

			m21 = _v2.m1;
			m22 = _v2.m2;

			m31 = _v3.m1;
			m32 = _v3.m2;

			m41 = _v4.m1;
			m42 = _v4.m2;
		}

			// Full scalar constructor
		inline dmat4(double  _d1, double  _d2, double  _d3, double  _d4,
					 double  _d5, double  _d6, double  _d7, double  _d8,
					 double  _d9, double _d10, double _d11, double _d12,
					 double _d13, double _d14, double _d15, double _d16) {
			m11 = _mm_setr_pd( _d1,  _d2);
			m12 = _mm_setr_pd( _d3,  _d4);

			m21 = _mm_setr_pd( _d5,  _d6);
			m22 = _mm_setr_pd( _d7,  _d8);

			m31 = _mm_setr_pd( _d9, _d10);
			m32 = _mm_setr_pd(_d11, _d12);

			m41 = _mm_setr_pd(_d13, _d14);
			m42 = _mm_setr_pd(_d15, _d16);
		}

			// Copy constructor
		inline dmat4(const dmat4 &m) {
			m11 = m.m11;
			m12 = m.m12;

			m21 = m.m21;
			m22 = m.m22;

			m31 = m.m31;
			m32 = m.m32;

			m41 = m.m41;
			m42 = m.m42;
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
		inline dvec4& operator[](int index) {
			return reinterpret_cast<dvec4 &>(m[index]);
		}

			// Read direct access operator
		inline const dvec4& operator[](int index) const {
			return reinterpret_cast<const dvec4 &>(m[index]);
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

		inline dmat4& operator += (double d) {
			__m128d dd = _mm_set1_pd(d);
			m11 = _mm_add_pd(m11, dd);
			m12 = _mm_add_pd(m12, dd);

			m21 = _mm_add_pd(m21, dd);
			m22 = _mm_add_pd(m22, dd);

			m31 = _mm_add_pd(m31, dd);
			m32 = _mm_add_pd(m32, dd);

			m41 = _mm_add_pd(m41, dd);
			m42 = _mm_add_pd(m42, dd);

			return *this;
		}

		inline dmat4& operator += (const dmat4 &m) {
			m11 = _mm_add_pd(m11, m.m11);
			m12 = _mm_add_pd(m12, m.m12);

			m21 = _mm_add_pd(m21, m.m21);
			m22 = _mm_add_pd(m22, m.m22);

			m31 = _mm_add_pd(m31, m.m31);
			m32 = _mm_add_pd(m32, m.m32);

			m41 = _mm_add_pd(m41, m.m41);
			m42 = _mm_add_pd(m42, m.m42);

			return *this;
		}

		inline dmat4& operator -= (double d) {
			__m128d dd = _mm_set1_pd(d);
			m11 = _mm_sub_pd(m11, dd);
			m12 = _mm_sub_pd(m12, dd);

			m21 = _mm_sub_pd(m21, dd);
			m22 = _mm_sub_pd(m22, dd);

			m31 = _mm_sub_pd(m31, dd);
			m32 = _mm_sub_pd(m32, dd);

			m41 = _mm_sub_pd(m41, dd);
			m42 = _mm_sub_pd(m42, dd);

			return *this;
		}

		inline dmat4& operator -= (const dmat4 &m) {
			m11 = _mm_sub_pd(m11, m.m11);
			m12 = _mm_sub_pd(m12, m.m12);

			m21 = _mm_sub_pd(m21, m.m21);
			m22 = _mm_sub_pd(m22, m.m22);

			m31 = _mm_sub_pd(m31, m.m31);
			m32 = _mm_sub_pd(m32, m.m32);

			m41 = _mm_sub_pd(m41, m.m41);
			m42 = _mm_sub_pd(m42, m.m42);

			return *this;
		}

		inline dmat4& operator *= (double d) {
			__m128d dd = _mm_set1_pd(d);
			m11 = _mm_mul_pd(m11, dd);
			m12 = _mm_mul_pd(m12, dd);

			m21 = _mm_mul_pd(m21, dd);
			m22 = _mm_mul_pd(m22, dd);

			m31 = _mm_mul_pd(m31, dd);
			m32 = _mm_mul_pd(m32, dd);

			m41 = _mm_mul_pd(m41, dd);
			m42 = _mm_mul_pd(m42, dd);

			return *this;
		}

		inline dmat4& operator *= (const dmat4 &m) {
			__m128d xx1 = _mm_unpacklo_pd(m11, m11);
			__m128d yy1 = _mm_unpackhi_pd(m11, m11);
			__m128d zz1 = _mm_unpacklo_pd(m12, m12);
			__m128d ww1 = _mm_unpackhi_pd(m12, m12);
			__m128d xx2 = _mm_unpacklo_pd(m21, m21);
			__m128d yy2 = _mm_unpackhi_pd(m21, m21);
			__m128d zz2 = _mm_unpacklo_pd(m22, m22);
			__m128d ww2 = _mm_unpackhi_pd(m22, m22);
			__m128d xx3 = _mm_unpacklo_pd(m31, m31);
			__m128d yy3 = _mm_unpackhi_pd(m31, m31);
			__m128d zz3 = _mm_unpacklo_pd(m32, m32);
			__m128d ww3 = _mm_unpackhi_pd(m32, m32);
			__m128d xx4 = _mm_unpacklo_pd(m41, m41);
			__m128d yy4 = _mm_unpackhi_pd(m41, m41);
			__m128d zz4 = _mm_unpacklo_pd(m42, m42);
			__m128d ww4 = _mm_unpackhi_pd(m42, m42);
			m11 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(m.m11, xx1),
										_mm_mul_pd(m.m21, yy1)),
							 _mm_add_pd(_mm_mul_pd(m.m31, zz1),
										_mm_mul_pd(m.m41, ww1)));
			m12 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(m.m12, xx1),
										_mm_mul_pd(m.m22, yy1)),
							 _mm_add_pd(_mm_mul_pd(m.m32, zz1),
										_mm_mul_pd(m.m42, ww1)));
			m21 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(m.m11, xx2),
										_mm_mul_pd(m.m21, yy2)),
							 _mm_add_pd(_mm_mul_pd(m.m31, zz2),
										_mm_mul_pd(m.m41, ww2)));
			m22 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(m.m12, xx2),
										_mm_mul_pd(m.m22, yy2)),
							 _mm_add_pd(_mm_mul_pd(m.m32, zz2),
										_mm_mul_pd(m.m42, ww2)));
			m31 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(m.m11, xx3),
										_mm_mul_pd(m.m21, yy3)),
							 _mm_add_pd(_mm_mul_pd(m.m31, zz3),
										_mm_mul_pd(m.m41, ww3)));
			m32 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(m.m12, xx3),
										_mm_mul_pd(m.m22, yy3)),
							 _mm_add_pd(_mm_mul_pd(m.m32, zz3),
										_mm_mul_pd(m.m42, ww3)));
			m41 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(m.m11, xx4),
										_mm_mul_pd(m.m21, yy4)),
							 _mm_add_pd(_mm_mul_pd(m.m31, zz4),
										_mm_mul_pd(m.m41, ww4)));
			m42 = _mm_add_pd(_mm_add_pd(_mm_mul_pd(m.m12, xx4),
										_mm_mul_pd(m.m22, yy4)),
							 _mm_add_pd(_mm_mul_pd(m.m32, zz4),
										_mm_mul_pd(m.m42, ww4)));

			return *this;
		}

		inline dmat4& operator /= (double d) {
			__m128d dd = _mm_set1_pd(d);
			m11 = _mm_div_pd(m11, dd);
			m12 = _mm_div_pd(m12, dd);

			m21 = _mm_div_pd(m21, dd);
			m22 = _mm_div_pd(m22, dd);

			m31 = _mm_div_pd(m31, dd);
			m32 = _mm_div_pd(m32, dd);

			m41 = _mm_div_pd(m41, dd);
			m42 = _mm_div_pd(m42, dd);

			return *this;
		}

		inline dmat4& operator /= (const dmat4 &m) {
			m11 = _mm_div_pd(m11, m.m11);
			m12 = _mm_div_pd(m12, m.m12);

			m21 = _mm_div_pd(m21, m.m21);
			m22 = _mm_div_pd(m22, m.m22);

			m31 = _mm_div_pd(m31, m.m31);
			m32 = _mm_div_pd(m32, m.m32);

			m41 = _mm_div_pd(m41, m.m41);
			m42 = _mm_div_pd(m42, m.m42);

			return *this;
		}

		// ----------------------------------------------------------------- //

		friend inline dmat4 operator + (const dmat4 &m, double d) {
			__m128d dd = _mm_set1_pd(d);
			return dmat4(_mm_add_pd(m.m11, dd), _mm_add_pd(m.m12, dd),
						 _mm_add_pd(m.m21, dd), _mm_add_pd(m.m22, dd),
						 _mm_add_pd(m.m31, dd), _mm_add_pd(m.m32, dd),
						 _mm_add_pd(m.m41, dd), _mm_add_pd(m.m42, dd));
		}

		friend inline dmat4 operator + (const dmat4 &m0, const dmat4 &m1) {
			return dmat4(_mm_add_pd(m0.m11, m1.m11), _mm_add_pd(m0.m12, m1.m12),
						 _mm_add_pd(m0.m21, m1.m21), _mm_add_pd(m0.m22, m1.m22),
						 _mm_add_pd(m0.m31, m1.m31), _mm_add_pd(m0.m32, m1.m32),
						 _mm_add_pd(m0.m41, m1.m41), _mm_add_pd(m0.m42, m1.m42));
		}

		friend inline dmat4 operator - (const dmat4 &m, double d) {
			__m128d dd = _mm_set1_pd(d);
			return dmat4(_mm_sub_pd(m.m11, dd), _mm_sub_pd(m.m12, dd),
						 _mm_sub_pd(m.m21, dd), _mm_sub_pd(m.m22, dd),
						 _mm_sub_pd(m.m31, dd), _mm_sub_pd(m.m32, dd),
						 _mm_sub_pd(m.m41, dd), _mm_sub_pd(m.m42, dd));
		}

		friend inline dmat4 operator - (double d, const dmat4 &m) {
			__m128d dd = _mm_set1_pd(d);
			return dmat4(_mm_sub_pd(dd, m.m11), _mm_sub_pd(dd, m.m12),
						 _mm_sub_pd(dd, m.m21), _mm_sub_pd(dd, m.m22),
						 _mm_sub_pd(dd, m.m31), _mm_sub_pd(dd, m.m32),
						 _mm_sub_pd(dd, m.m41), _mm_sub_pd(dd, m.m42));
		}

		friend inline dmat4 operator - (const dmat4 &m0, const dmat4 &m1) {
			return dmat4(_mm_sub_pd(m0.m11, m1.m11), _mm_sub_pd(m0.m12, m1.m12),
						 _mm_sub_pd(m0.m21, m1.m21), _mm_sub_pd(m0.m22, m1.m22),
						 _mm_sub_pd(m0.m31, m1.m31), _mm_sub_pd(m0.m32, m1.m32),
						 _mm_sub_pd(m0.m41, m1.m41), _mm_sub_pd(m0.m42, m1.m42));
		}

		friend inline dmat4 operator * (const dmat4 &m, double d) {
			__m128d dd = _mm_set1_pd(d);
			return dmat4(_mm_mul_pd(m.m11, dd), _mm_mul_pd(m.m12, dd),
						 _mm_mul_pd(m.m21, dd), _mm_mul_pd(m.m22, dd),
						 _mm_mul_pd(m.m31, dd), _mm_mul_pd(m.m32, dd),
						 _mm_mul_pd(m.m41, dd), _mm_mul_pd(m.m42, dd));
		}

		friend inline dvec4 operator * (const dmat4 &m, const dvec4 &v) {
			__m128d _xx = _mm_unpacklo_pd(v.m1, v.m1);
			__m128d _yy = _mm_unpackhi_pd(v.m1, v.m1);
			__m128d _zz = _mm_unpacklo_pd(v.m2, v.m2);
			__m128d _ww = _mm_unpackhi_pd(v.m2, v.m2);
			return dvec4(_mm_add_pd(_mm_add_pd(_mm_mul_pd(m.m11, _xx),
											   _mm_mul_pd(m.m21, _yy)),
									_mm_add_pd(_mm_mul_pd(m.m31, _zz),
											   _mm_mul_pd(m.m41, _ww))),
						 _mm_add_pd(_mm_add_pd(_mm_mul_pd(m.m12, _xx),
											   _mm_mul_pd(m.m22, _yy)),
									_mm_add_pd(_mm_mul_pd(m.m32, _zz),
											   _mm_mul_pd(m.m42, _ww))));
		}

		friend inline dvec4 operator * (const dvec4 &v, const dmat4 &m) {
			__m128d _xx = _mm_unpacklo_pd(v.m1, v.m1);
			__m128d _yy = _mm_unpackhi_pd(v.m1, v.m1);
			__m128d _zz = _mm_unpacklo_pd(v.m2, v.m2);
			__m128d _ww = _mm_unpackhi_pd(v.m2, v.m2);
			return dvec4(_mm_add_pd(_mm_add_pd(_mm_mul_pd(_mm_unpacklo_pd(m.m11, m.m21), _xx),
											   _mm_mul_pd(_mm_unpackhi_pd(m.m11, m.m21), _yy)),
									_mm_add_pd(_mm_mul_pd(_mm_unpacklo_pd(m.m12, m.m22), _zz),
											   _mm_mul_pd(_mm_unpackhi_pd(m.m12, m.m22), _ww))),
						 _mm_add_pd(_mm_add_pd(_mm_mul_pd(_mm_unpacklo_pd(m.m31, m.m41), _xx),
											   _mm_mul_pd(_mm_unpackhi_pd(m.m31, m.m41), _yy)),
									_mm_add_pd(_mm_mul_pd(_mm_unpacklo_pd(m.m32, m.m42), _zz),
											   _mm_mul_pd(_mm_unpackhi_pd(m.m32, m.m42), _ww))));
		}

		friend inline dmat4 operator * (const dmat4 &m0, const dmat4 &m1) {
			__m128d xx1 = _mm_unpacklo_pd(m0[0].m1, m0[0].m1);
			__m128d yy1 = _mm_unpackhi_pd(m0[0].m1, m0[0].m1);
			__m128d zz1 = _mm_unpacklo_pd(m0[0].m2, m0[0].m2);
			__m128d ww1 = _mm_unpackhi_pd(m0[0].m2, m0[0].m2);
			__m128d xx2 = _mm_unpacklo_pd(m0[1].m1, m0[1].m1);
			__m128d yy2 = _mm_unpackhi_pd(m0[1].m1, m0[1].m1);
			__m128d zz2 = _mm_unpacklo_pd(m0[1].m2, m0[1].m2);
			__m128d ww2 = _mm_unpackhi_pd(m0[1].m2, m0[1].m2);
			__m128d xx3 = _mm_unpacklo_pd(m0[2].m1, m0[2].m1);
			__m128d yy3 = _mm_unpackhi_pd(m0[2].m1, m0[2].m1);
			__m128d zz3 = _mm_unpacklo_pd(m0[2].m2, m0[2].m2);
			__m128d ww3 = _mm_unpackhi_pd(m0[2].m2, m0[2].m2);
			__m128d xx4 = _mm_unpacklo_pd(m0[3].m1, m0[3].m1);
			__m128d yy4 = _mm_unpackhi_pd(m0[3].m1, m0[3].m1);
			__m128d zz4 = _mm_unpacklo_pd(m0[3].m2, m0[3].m2);
			__m128d ww4 = _mm_unpackhi_pd(m0[3].m2, m0[3].m2);
			return dmat4(_mm_add_pd(_mm_add_pd(_mm_mul_pd(m1.m11, xx1),
											   _mm_mul_pd(m1.m21, yy1)),
									_mm_add_pd(_mm_mul_pd(m1.m31, zz1),
											   _mm_mul_pd(m1.m41, ww1))),
						 _mm_add_pd(_mm_add_pd(_mm_mul_pd(m1.m12, xx1),
											   _mm_mul_pd(m1.m22, yy1)),
									_mm_add_pd(_mm_mul_pd(m1.m32, zz1),
											   _mm_mul_pd(m1.m42, ww1))),
						 _mm_add_pd(_mm_add_pd(_mm_mul_pd(m1.m11, xx2),
											   _mm_mul_pd(m1.m21, yy2)),
									_mm_add_pd(_mm_mul_pd(m1.m31, zz2),
											   _mm_mul_pd(m1.m41, ww2))),
						 _mm_add_pd(_mm_add_pd(_mm_mul_pd(m1.m12, xx2),
											   _mm_mul_pd(m1.m22, yy2)),
									_mm_add_pd(_mm_mul_pd(m1.m32, zz2),
											   _mm_mul_pd(m1.m42, ww2))),
						 _mm_add_pd(_mm_add_pd(_mm_mul_pd(m1.m11, xx3),
											   _mm_mul_pd(m1.m21, yy3)),
									_mm_add_pd(_mm_mul_pd(m1.m31, zz3),
											   _mm_mul_pd(m1.m41, ww3))),
						 _mm_add_pd(_mm_add_pd(_mm_mul_pd(m1.m12, xx3),
											   _mm_mul_pd(m1.m22, yy3)),
									_mm_add_pd(_mm_mul_pd(m1.m32, zz3),
											   _mm_mul_pd(m1.m42, ww3))),
						 _mm_add_pd(_mm_add_pd(_mm_mul_pd(m1.m11, xx4),
											   _mm_mul_pd(m1.m21, yy4)),
									_mm_add_pd(_mm_mul_pd(m1.m31, zz4),
											   _mm_mul_pd(m1.m41, ww4))),
						 _mm_add_pd(_mm_add_pd(_mm_mul_pd(m1.m12, xx4),
											   _mm_mul_pd(m1.m22, yy4)),
									_mm_add_pd(_mm_mul_pd(m1.m32, zz4),
											   _mm_mul_pd(m1.m42, ww4))));
		}

		friend inline dmat4 operator / (const dmat4 &m, double d) {
			__m128d dd = _mm_set1_pd(d);
			return dmat4(_mm_div_pd(m.m11, dd), _mm_div_pd(m.m12, dd),
						 _mm_div_pd(m.m21, dd), _mm_div_pd(m.m22, dd),
						 _mm_div_pd(m.m31, dd), _mm_div_pd(m.m32, dd),
						 _mm_div_pd(m.m41, dd), _mm_div_pd(m.m42, dd));
		}

		friend inline dmat4 operator / (double d, const dmat4 &m) {
			__m128d dd = _mm_set1_pd(d);
			return dmat4(_mm_div_pd(dd, m.m11), _mm_div_pd(dd, m.m12),
						 _mm_div_pd(dd, m.m21), _mm_div_pd(dd, m.m22),
						 _mm_div_pd(dd, m.m31), _mm_div_pd(dd, m.m32),
						 _mm_div_pd(dd, m.m41), _mm_div_pd(dd, m.m42));
		}

		friend inline dmat4 operator / (const dmat4 &m0, const dmat4 &m1) {
			return dmat4(_mm_div_pd(m0.m11, m1.m11), _mm_div_pd(m0.m12, m1.m12),
						 _mm_div_pd(m0.m21, m1.m21), _mm_div_pd(m0.m22, m1.m22),
						 _mm_div_pd(m0.m31, m1.m31), _mm_div_pd(m0.m32, m1.m32),
						 _mm_div_pd(m0.m41, m1.m41), _mm_div_pd(m0.m42, m1.m42));
		}

		// ----------------------------------------------------------------- //

		friend inline dmat4 matrixCompMult(const dmat4 &m0, const dmat4 &m1) {
			return dmat4(_mm_mul_pd(m0.m11, m1.m11), _mm_mul_pd(m0.m12, m1.m12),
						 _mm_mul_pd(m0.m21, m1.m21), _mm_mul_pd(m0.m22, m1.m22),
						 _mm_mul_pd(m0.m31, m1.m31), _mm_mul_pd(m0.m32, m1.m32),
						 _mm_mul_pd(m0.m41, m1.m41), _mm_mul_pd(m0.m42, m1.m42));
		}

		// ----------------------------------------------------------------- //

		friend inline dmat4 transpose(const dmat4 &m) {
			return dmat4(_mm_unpacklo_pd(m.m11, m.m21), _mm_unpacklo_pd(m.m31, m.m41),
						 _mm_unpackhi_pd(m.m11, m.m21), _mm_unpackhi_pd(m.m31, m.m41),
						 _mm_unpacklo_pd(m.m12, m.m22), _mm_unpacklo_pd(m.m32, m.m42),
						 _mm_unpackhi_pd(m.m12, m.m22), _mm_unpackhi_pd(m.m32, m.m42));
		}

		friend inline double determinant(const dmat4 &m) {
			__m128d r1 = _mm_mul_pd(m.m11, _mm_shuffle_pd(m.m21, m.m21, 0x01));
			__m128d r2 = _mm_mul_pd(m.m12, _mm_shuffle_pd(m.m22, m.m22, 0x01));
			__m128d r3 = _mm_mul_pd(m.m31, _mm_shuffle_pd(m.m41, m.m41, 0x01));
			__m128d r4 = _mm_mul_pd(m.m32, _mm_shuffle_pd(m.m42, m.m42, 0x01));
			__m128d c1 = _mm_sub_pd(_mm_mul_pd(m.m31, _mm_unpackhi_pd(m.m42, m.m42)),
									_mm_mul_pd(m.m41, _mm_unpackhi_pd(m.m32, m.m32)));
			__m128d c2 = _mm_sub_pd(_mm_mul_pd(m.m41, _mm_unpacklo_pd(m.m32, m.m32)),
									_mm_mul_pd(m.m31, _mm_unpacklo_pd(m.m42, m.m42)));
			__m128d d  = _mm_add_pd(_mm_mul_pd(_mm_sub_pd(
									_mm_mul_pd(m.m12, _mm_unpackhi_pd(m.m21, m.m21)),
									_mm_mul_pd(m.m22, _mm_unpackhi_pd(m.m11, m.m11))),
													  _mm_unpacklo_pd(c1, c2)),
								    _mm_mul_pd(_mm_sub_pd(
								    _mm_mul_pd(m.m22, _mm_unpacklo_pd(m.m11, m.m11)),
									_mm_mul_pd(m.m12, _mm_unpacklo_pd(m.m21, m.m21))),
													  _mm_unpackhi_pd(c1, c2)));
			r1 = _mm_sub_sd(r1, _mm_unpackhi_pd(r1, r1));
			r2 = _mm_sub_sd(r2, _mm_unpackhi_pd(r2, r2));
			r3 = _mm_sub_sd(r3, _mm_unpackhi_pd(r3, r3));
			r4 = _mm_sub_sd(r4, _mm_unpackhi_pd(r4, r4));
			return _mm_cvtsd_f64(_mm_sub_sd(_mm_add_sd(_mm_mul_sd(r1, r4),
													   _mm_mul_sd(r2, r3)),
											_mm_add_sd(_mm_unpackhi_pd(d, d), d)));
		}

		friend inline dmat4 inverse(const dmat4 &m) {
			__m128d r1 = _mm_mul_pd(m.m11, _mm_shuffle_pd(m.m21, m.m21, 0x01));
			__m128d r2 = _mm_mul_pd(m.m12, _mm_shuffle_pd(m.m22, m.m22, 0x01));
			__m128d r3 = _mm_mul_pd(m.m31, _mm_shuffle_pd(m.m41, m.m41, 0x01));
			__m128d r4 = _mm_mul_pd(m.m32, _mm_shuffle_pd(m.m42, m.m42, 0x01));
			__m128d v11 = _mm_sub_pd(_mm_mul_pd(_mm_unpackhi_pd(m.m21, m.m21), m.m12),
									 _mm_mul_pd(_mm_unpackhi_pd(m.m11, m.m11), m.m22));
			__m128d v12 = _mm_sub_pd(_mm_mul_pd(_mm_unpacklo_pd(m.m11, m.m11), m.m22),
									 _mm_mul_pd(_mm_unpacklo_pd(m.m21, m.m21), m.m12));
			__m128d v21 = _mm_sub_pd(_mm_mul_pd(_mm_unpackhi_pd(m.m42, m.m42), m.m31),
									 _mm_mul_pd(_mm_unpackhi_pd(m.m32, m.m32), m.m41));
			__m128d v22 = _mm_sub_pd(_mm_mul_pd(_mm_unpacklo_pd(m.m32, m.m32), m.m41),
									 _mm_mul_pd(_mm_unpacklo_pd(m.m42, m.m42), m.m31));
			__m128d d   = _mm_add_pd(_mm_mul_pd(v11, _mm_unpacklo_pd(v21, v22)),
								     _mm_mul_pd(v12, _mm_unpackhi_pd(v21, v22)));
			r1 = _mm_sub_sd(r1, _mm_unpackhi_pd(r1, r1));
			r2 = _mm_sub_sd(r2, _mm_unpackhi_pd(r2, r2));
			r3 = _mm_sub_sd(r3, _mm_unpackhi_pd(r3, r3));
			r4 = _mm_sub_sd(r4, _mm_unpackhi_pd(r4, r4));
			d = _mm_add_sd(_mm_unpackhi_pd(d, d), d);
			d = _mm_div_sd(_mm_set_sd(1.0),
						   _mm_sub_sd(_mm_add_sd(_mm_mul_sd(r1, r4),
												 _mm_mul_sd(r2, r3)), d));
			d  = _mm_unpacklo_pd( d,  d);
			r1 = _mm_unpacklo_pd(r1, r1);
			r2 = _mm_unpacklo_pd(r2, r2);
			r3 = _mm_unpacklo_pd(r3, r3);
			r4 = _mm_unpacklo_pd(r4, r4);
			__m128d i11 = _mm_sub_pd(_mm_mul_pd(m.m11, r4), _mm_add_pd(
									 _mm_mul_pd(v21, _mm_unpacklo_pd(m.m12, m.m12)),
									 _mm_mul_pd(v22, _mm_unpackhi_pd(m.m12, m.m12))));
			__m128d i12 = _mm_sub_pd(_mm_mul_pd(m.m21, r4), _mm_add_pd(
									 _mm_mul_pd(v21, _mm_unpacklo_pd(m.m22, m.m22)),
									 _mm_mul_pd(v22, _mm_unpackhi_pd(m.m22, m.m22))));
			__m128d i41 = _mm_sub_pd(_mm_mul_pd(m.m32, r1), _mm_add_pd(
									 _mm_mul_pd(v11, _mm_unpacklo_pd(m.m31, m.m31)),
									 _mm_mul_pd(v12, _mm_unpackhi_pd(m.m31, m.m31))));
			__m128d i42 = _mm_sub_pd(_mm_mul_pd(m.m42, r1), _mm_add_pd(
									 _mm_mul_pd(v11, _mm_unpacklo_pd(m.m41, m.m41)),
									 _mm_mul_pd(v12, _mm_unpackhi_pd(m.m41, m.m41))));
			__m128d i21 = _mm_sub_pd(_mm_mul_pd(m.m31, r2), _mm_sub_pd(
									 _mm_mul_pd(_mm_shuffle_pd(v12, v11, 0x01), m.m32),
									 _mm_mul_pd(_mm_shuffle_pd(m.m32, m.m32, 0x01),
												_mm_shuffle_pd(v12, v11, 0x02))));
			__m128d i22 = _mm_sub_pd(_mm_mul_pd(m.m41, r2), _mm_sub_pd(
									 _mm_mul_pd(_mm_shuffle_pd(v12, v11, 0x01), m.m42),
									 _mm_mul_pd(_mm_shuffle_pd(m.m42, m.m42, 0x01),
												_mm_shuffle_pd(v12, v11, 0x02))));
			__m128d i31 = _mm_sub_pd(_mm_mul_pd(m.m12, r3), _mm_sub_pd(
									 _mm_mul_pd(_mm_shuffle_pd(v22, v21, 0x01), m.m11),
									 _mm_mul_pd(_mm_shuffle_pd(m.m11, m.m11, 0x01),
												_mm_shuffle_pd(v22, v21, 0x02))));
			__m128d i32 = _mm_sub_pd(_mm_mul_pd(m.m22, r3), _mm_sub_pd(
									 _mm_mul_pd(_mm_shuffle_pd(v22, v21, 0x01), m.m21),
									 _mm_mul_pd(_mm_shuffle_pd(m.m21, m.m21, 0x01),
												_mm_shuffle_pd(v22, v21, 0x02))));
			__m128d d1 = _mm_xor_pd(d, _mm_setr_pd( 0.0, -0.0));
			__m128d d2 = _mm_xor_pd(d, _mm_setr_pd(-0.0,  0.0));
			return dmat4(_mm_mul_pd(_mm_unpackhi_pd(i12, i11), d1),
						 _mm_mul_pd(_mm_unpackhi_pd(i22, i21), d1),
						 _mm_mul_pd(_mm_unpacklo_pd(i12, i11), d2),
						 _mm_mul_pd(_mm_unpacklo_pd(i22, i21), d2),
						 _mm_mul_pd(_mm_unpackhi_pd(i32, i31), d1),
						 _mm_mul_pd(_mm_unpackhi_pd(i42, i41), d1),
						 _mm_mul_pd(_mm_unpacklo_pd(i32, i31), d2),
						 _mm_mul_pd(_mm_unpacklo_pd(i42, i41), d2));
		}

		// ----------------------------------------------------------------- //

	private:
			// SSE constructor
		inline dmat4(const __m128d &_m11, const __m128d &_m12,
					 const __m128d &_m21, const __m128d &_m22,
					 const __m128d &_m31, const __m128d &_m32,
					 const __m128d &_m41, const __m128d &_m42) {
			m11 = _m11;
			m12 = _m12;

			m21 = _m21;
			m22 = _m22;

			m31 = _m31;
			m32 = _m32;

			m41 = _m41;
			m42 = _m42;
		}

		union {
			__m128d m[4][2];
			struct {
				__m128d m11, m12;
				__m128d m21, m22;
				__m128d m31, m32;
				__m128d m41, m42;
			};

/*			// This code is waiting for unrestricted unions feature in c++0x
			dvec4 v[4];
			struct {
				dvec4 v1;
				dvec4 v2;
				dvec4 v3;
				dvec4 v4;
			};
*/
		};
};

#endif
