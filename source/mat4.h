#include "vec4.h"

#ifndef __MAT4_H__
#define __MAT4_H__

extern void printv(const vec4&);

class mat4
{
	private:
			// Most compilers don't use pshufd (SSE2) when _mm_shuffle(x, x, mask) is used
			// This macro saves 2-3 movaps instructions when shuffling
			// This has to be a macro since mask HAS to be an immidiate value
		#define _mm_shufd(xmm, mask) _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(xmm), mask))

	public:
			// Identity matrix
		inline mat4() {
			m1 = _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f);
			m2 = _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f);
			m3 = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
			m4 = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
		}

			// Scaled matrix
		explicit inline mat4(float f) {
			m1 = _mm_setr_ps(   f, 0.0f, 0.0f, 0.0f);
			m2 = _mm_setr_ps(0.0f,    f, 0.0f, 0.0f);
			m3 = _mm_setr_ps(0.0f, 0.0f,    f, 0.0f);
			m4 = _mm_setr_ps(0.0f, 0.0f, 0.0f,    f);
		}
			
			// 4 vectors constructor
		inline mat4(const vec4 &_v1, const vec4 &_v2,
		            const vec4 &_v3, const vec4 &_v4) {
			m1 = _v1.m;
			m2 = _v2.m;
			m3 = _v3.m;
			m4 = _v4.m;
		}
			
			// Full scalar constructor
		inline mat4(float  _f1, float  _f2, float  _f3, float  _f4, 
		            float  _f5, float  _f6, float  _f7, float  _f8,
		            float  _f9, float _f10, float _f11, float _f12,
		            float _f13, float _f14, float _f15, float _f16) {
			m1 = _mm_setr_ps( _f1,  _f2,  _f3,  _f4);
			m2 = _mm_setr_ps( _f5,  _f6,  _f7,  _f8);
			m3 = _mm_setr_ps( _f9, _f10, _f11, _f12);
			m4 = _mm_setr_ps(_f13, _f14, _f15, _f16);
		}
			
			// Copy constructor
		inline mat4(const mat4 &m) {
			m1 = m.m1;
			m2 = m.m2;
			m3 = m.m3;
			m4 = m.m4;
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
		inline vec4& operator[](int index) {
			return reinterpret_cast<vec4 &>(m[index]);
		}

			// Read direct access operator
		inline const vec4& operator[](int index) const {
			return reinterpret_cast<const vec4 &>(m[index]);
		}

			// Cast operator
		inline operator float*() {
			return reinterpret_cast<float *>(this);
		}

			// Const cast operator
		inline operator const float*() const {
			return reinterpret_cast<const float *>(this);
		}

		// ----------------------------------------------------------------- //

		inline mat4& operator += (float f) {
			__m128 ff = _mm_set1_ps(f);
			m1 = _mm_add_ps(m1, ff);
			m2 = _mm_add_ps(m2, ff);
			m3 = _mm_add_ps(m3, ff);
			m4 = _mm_add_ps(m4, ff);

			return *this;
		}

		inline mat4& operator += (const mat4 &m) {
			m1 = _mm_add_ps(m1, m.m1);
			m2 = _mm_add_ps(m2, m.m2);
			m3 = _mm_add_ps(m3, m.m3);
			m4 = _mm_add_ps(m4, m.m4);

			return *this;
		}

		inline mat4& operator -= (float f) {
			__m128 ff = _mm_set1_ps(f);
			m1 = _mm_sub_ps(m1, ff);
			m2 = _mm_sub_ps(m2, ff);
			m3 = _mm_sub_ps(m3, ff);
			m4 = _mm_sub_ps(m4, ff);

			return *this;
		}

		inline mat4& operator -= (const mat4 &m) {
			m1 = _mm_sub_ps(m1, m.m1);
			m2 = _mm_sub_ps(m2, m.m2);
			m3 = _mm_sub_ps(m3, m.m3);
			m4 = _mm_sub_ps(m4, m.m4);

			return *this;
		}

		inline mat4& operator *= (float f) {
			__m128 ff = _mm_set1_ps(f);
			m1 = _mm_mul_ps(m1, ff);
			m2 = _mm_mul_ps(m2, ff);
			m3 = _mm_mul_ps(m3, ff);
			m4 = _mm_mul_ps(m4, ff);

			return *this;
		}

		inline mat4& operator *= (const mat4 &m) {
			m1 = _mm_add_ps(_mm_add_ps(
							_mm_add_ps(_mm_mul_ps(_mm_shufd(m1, 0x00), m.m1),
									   _mm_mul_ps(_mm_shufd(m1, 0x55), m.m2)),
									   _mm_mul_ps(_mm_shufd(m1, 0xAA), m.m3)),
									   _mm_mul_ps(_mm_shufd(m1, 0xFF), m.m4));
			m2 = _mm_add_ps(_mm_add_ps(
							_mm_add_ps(_mm_mul_ps(_mm_shufd(m2, 0x00), m.m1),
									   _mm_mul_ps(_mm_shufd(m2, 0x55), m.m2)),
									   _mm_mul_ps(_mm_shufd(m2, 0xAA), m.m3)),
									   _mm_mul_ps(_mm_shufd(m2, 0xFF), m.m4));
			m3 = _mm_add_ps(_mm_add_ps(
							_mm_add_ps(_mm_mul_ps(_mm_shufd(m3, 0x00), m.m1),
									   _mm_mul_ps(_mm_shufd(m3, 0x55), m.m2)),
									   _mm_mul_ps(_mm_shufd(m3, 0xAA), m.m3)),
									   _mm_mul_ps(_mm_shufd(m3, 0xFF), m.m4));
			m4 = _mm_add_ps(_mm_add_ps(
							_mm_add_ps(_mm_mul_ps(_mm_shufd(m4, 0x00), m.m1),
									   _mm_mul_ps(_mm_shufd(m4, 0x55), m.m2)),
									   _mm_mul_ps(_mm_shufd(m4, 0xAA), m.m3)),
									   _mm_mul_ps(_mm_shufd(m4, 0xFF), m.m4));
			return *this;
		}

		inline mat4& operator /= (float f) {
			__m128 ff = _mm_set1_ps(f);
			m1 = _mm_div_ps(m1, ff);
			m2 = _mm_div_ps(m2, ff);
			m3 = _mm_div_ps(m3, ff);
			m4 = _mm_div_ps(m4, ff);

			return *this;
		}

		inline mat4& operator /= (const mat4 &m) {
			m1 = _mm_div_ps(m1, m.m1);
			m2 = _mm_div_ps(m2, m.m2);
			m3 = _mm_div_ps(m3, m.m3);
			m4 = _mm_div_ps(m4, m.m4);

			return *this;
		}

		// ----------------------------------------------------------------- //

		friend inline mat4 operator + (const mat4 &m, float f) {
			__m128 ff = _mm_set1_ps(f);
			return mat4(_mm_add_ps(m.m1, ff), _mm_add_ps(m.m2, ff),
						_mm_add_ps(m.m3, ff), _mm_add_ps(m.m4, ff));
		}

		friend inline mat4 operator + (const mat4 &m0, const mat4 &m1) {
			return mat4(_mm_add_ps(m0.m1, m1.m1), _mm_add_ps(m0.m2, m1.m2),
						_mm_add_ps(m0.m3, m1.m3), _mm_add_ps(m0.m4, m1.m4));
		}

		friend inline mat4 operator - (const mat4 &m, float f) {
			__m128 ff = _mm_set1_ps(f);
			return mat4(_mm_sub_ps(m.m1, ff), _mm_sub_ps(m.m2, ff),
						_mm_sub_ps(m.m3, ff), _mm_sub_ps(m.m4, ff));
		}

		friend inline mat4 operator - (float f, const mat4 &m) {
			__m128 ff = _mm_set1_ps(f);
			return mat4(_mm_sub_ps(ff, m.m1), _mm_sub_ps(ff, m.m2),
						_mm_sub_ps(ff, m.m3), _mm_sub_ps(ff, m.m4));
		}

		friend inline mat4 operator - (const mat4 &m0, const mat4 &m1) {
			return mat4(_mm_sub_ps(m0.m1, m1.m1), _mm_sub_ps(m0.m2, m1.m2),
						_mm_sub_ps(m0.m3, m1.m3), _mm_sub_ps(m0.m4, m1.m4));
		}

		friend inline mat4 operator * (const mat4 &m, float f) {
			__m128 ff = _mm_set1_ps(f);
			return mat4(_mm_mul_ps(m.m1, ff), _mm_mul_ps(m.m2, ff),
						_mm_mul_ps(m.m3, ff), _mm_mul_ps(m.m4, ff));
		}

		friend inline vec4 operator * (const mat4 &m, const vec4 &v) {
			return _mm_add_ps(_mm_add_ps(
							  _mm_mul_ps(m.m1, _mm_shufd(v.m, 0x00)),
							  _mm_mul_ps(m.m2, _mm_shufd(v.m, 0x55))),
				   _mm_add_ps(_mm_mul_ps(m.m3, _mm_shufd(v.m, 0xAA)),
							  _mm_mul_ps(m.m4, _mm_shufd(v.m, 0xFF))));
		}

		friend inline vec4 operator * (const vec4 &v, const mat4 &m) {
			__m128 t1 = _mm_unpacklo_ps(m.m1, m.m2);
			__m128 t2 = _mm_unpacklo_ps(m.m3, m.m4);
			__m128 t3 = _mm_unpackhi_ps(m.m1, m.m2);
			__m128 t4 = _mm_unpackhi_ps(m.m3, m.m4);
			return _mm_add_ps(_mm_add_ps(
							  _mm_mul_ps(_mm_movelh_ps(t1, t2),
										 _mm_shufd(v.m, 0x00)),
							  _mm_mul_ps(_mm_movehl_ps(t2, t1),
										 _mm_shufd(v.m, 0x55))),
				   _mm_add_ps(_mm_mul_ps(_mm_movelh_ps(t3, t4),
									     _mm_shufd(v.m, 0xAA)),
							  _mm_mul_ps(_mm_movehl_ps(t4, t3),
										 _mm_shufd(v.m, 0xFF))));
		}

		friend inline mat4 operator * (const mat4 &m0, const mat4 &m1) {
			return mat4(_mm_add_ps(_mm_add_ps(
								   _mm_add_ps(_mm_mul_ps(_mm_shufd(m0.m1, 0x00), m1.m1),
											  _mm_mul_ps(_mm_shufd(m0.m1, 0x55), m1.m2)),
											  _mm_mul_ps(_mm_shufd(m0.m1, 0xAA), m1.m3)),
											  _mm_mul_ps(_mm_shufd(m0.m1, 0xFF), m1.m4)),
						_mm_add_ps(_mm_add_ps(
								   _mm_add_ps(_mm_mul_ps(_mm_shufd(m0.m2, 0x00), m1.m1),
											  _mm_mul_ps(_mm_shufd(m0.m2, 0x55), m1.m2)),
											  _mm_mul_ps(_mm_shufd(m0.m2, 0xAA), m1.m3)),
											  _mm_mul_ps(_mm_shufd(m0.m2, 0xFF), m1.m4)),
						_mm_add_ps(_mm_add_ps(
								   _mm_add_ps(_mm_mul_ps(_mm_shufd(m0.m3, 0x00), m1.m1),
											  _mm_mul_ps(_mm_shufd(m0.m3, 0x55), m1.m2)),
											  _mm_mul_ps(_mm_shufd(m0.m3, 0xAA), m1.m3)),
											  _mm_mul_ps(_mm_shufd(m0.m3, 0xFF), m1.m4)),
						_mm_add_ps(_mm_add_ps(
								   _mm_add_ps(_mm_mul_ps(_mm_shufd(m0.m4, 0x00), m1.m1),
											  _mm_mul_ps(_mm_shufd(m0.m4, 0x55), m1.m2)),
											  _mm_mul_ps(_mm_shufd(m0.m4, 0xAA), m1.m3)),
											  _mm_mul_ps(_mm_shufd(m0.m4, 0xFF), m1.m4)));
		}

		friend inline mat4 operator / (const mat4 &m, float f) {
			__m128 ff = _mm_set1_ps(f);
			return mat4(_mm_div_ps(m.m1, ff), _mm_div_ps(m.m2, ff),
						_mm_div_ps(m.m3, ff), _mm_div_ps(m.m4, ff));
		}

		friend inline mat4 operator / (float f, const mat4 &m) {
			__m128 ff = _mm_set1_ps(f);
			return mat4(_mm_div_ps(ff, m.m1), _mm_div_ps(ff, m.m2),
						_mm_div_ps(ff, m.m3), _mm_div_ps(ff, m.m4));
		}

		friend inline mat4 operator / (const mat4 &m0, const mat4 &m1) {
			return mat4(_mm_div_ps(m0.m1, m1.m1), _mm_div_ps(m0.m2, m1.m2),
						_mm_div_ps(m0.m3, m1.m3), _mm_div_ps(m0.m4, m1.m4));
		}

		// ----------------------------------------------------------------- //

		friend inline mat4 matrixCompMult(const mat4 &m0, const mat4 &m1) {
			return mat4(_mm_mul_ps(m0.m1, m1.m1), _mm_mul_ps(m0.m2, m1.m2),
						_mm_mul_ps(m0.m3, m1.m3), _mm_mul_ps(m0.m4, m1.m4));
		}

		// ----------------------------------------------------------------- //

		friend inline mat4 transpose(const mat4 &m) {
			__m128 t1 = _mm_unpacklo_ps(m.m1, m.m2);
			__m128 t2 = _mm_unpacklo_ps(m.m3, m.m4);
			__m128 t3 = _mm_unpackhi_ps(m.m1, m.m2);
			__m128 t4 = _mm_unpackhi_ps(m.m3, m.m4);
			return mat4(_mm_movelh_ps(t1, t2), _mm_movehl_ps(t2, t1),
						_mm_movelh_ps(t3, t4), _mm_movehl_ps(t4, t3));
		}

		friend inline float determinant(const mat4 &m) {
			__m128 r  = 			  _mm_shufd(m.m3, 0x39 );
			__m128 v1 = _mm_mul_ps(r,           m.m4       );
			__m128 v2 = _mm_mul_ps(r, _mm_shufd(m.m4, 0x4E));
			__m128 v3 = _mm_mul_ps(r, _mm_shufd(m.m4, 0x93));
			__m128 r1 = _mm_sub_ps(_mm_shufd(v2, 0x39),
								   _mm_shufd(v1, 0x4E));
			__m128 r2 = _mm_sub_ps(_mm_shufd(v3, 0x4E), v3);
			__m128 r3 = _mm_sub_ps(v2, _mm_shufd(v1, 0x39));

			v1 = _mm_shufd(m.m2, 0x93);
			v2 = _mm_shufd(m.m2, 0x39);
			v3 = _mm_shufd(m.m2, 0x4E);
			__m128 d = _mm_mul_ps(_mm_add_ps(_mm_add_ps(
											 _mm_mul_ps(v2, r1),
											 _mm_mul_ps(v3, r2)),
											 _mm_mul_ps(v1, r3)), m.m1);
			d = _mm_add_ps(d, _mm_shufd(d, 0x4E));
			d = _mm_sub_ss(d, _mm_shufd(d, 0x11));
			return _mm_cvtss_f32(d);
		}

		friend inline mat4 invert(const mat4 &m) {
			__m128 f1 = _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(m.m[2], m.m[1], 0xAA),
									_mm_shufd(_mm_shuffle_ps(m.m[3], m.m[2], 0xFF), 0x80)),
						 _mm_mul_ps(_mm_shufd(_mm_shuffle_ps(m.m[3], m.m[2], 0xAA), 0x80),
											  _mm_shuffle_ps(m.m[2], m.m[1], 0xFF)));
			__m128 f2 = _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(m.m[2], m.m[1], 0x55),
									_mm_shufd(_mm_shuffle_ps(m.m[3], m.m[2], 0xFF), 0x80)),
						 _mm_mul_ps(_mm_shufd(_mm_shuffle_ps(m.m[3], m.m[2], 0x55), 0x80),
											  _mm_shuffle_ps(m.m[2], m.m[1], 0xFF)));
			__m128 f3 = _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(m.m[2], m.m[1], 0x55),
									_mm_shufd(_mm_shuffle_ps(m.m[3], m.m[2], 0xAA), 0x80)),
						 _mm_mul_ps(_mm_shufd(_mm_shuffle_ps(m.m[3], m.m[2], 0x55), 0x80),
											  _mm_shuffle_ps(m.m[2], m.m[1], 0xAA)));
			__m128 f4 = _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(m.m[2], m.m[1], 0x00),
									_mm_shufd(_mm_shuffle_ps(m.m[3], m.m[2], 0xFF), 0x80)),
						 _mm_mul_ps(_mm_shufd(_mm_shuffle_ps(m.m[3], m.m[2], 0x00), 0x80),
											  _mm_shuffle_ps(m.m[2], m.m[1], 0xFF)));
			__m128 f5 = _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(m.m[2], m.m[1], 0x00),
									_mm_shufd(_mm_shuffle_ps(m.m[3], m.m[2], 0xAA), 0x80)),
						 _mm_mul_ps(_mm_shufd(_mm_shuffle_ps(m.m[3], m.m[2], 0x00), 0x80),
											  _mm_shuffle_ps(m.m[2], m.m[1], 0xAA)));
			__m128 f6 = _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(m.m[2], m.m[1], 0x00),
									_mm_shufd(_mm_shuffle_ps(m.m[3], m.m[2], 0x55), 0x80)),
						 _mm_mul_ps(_mm_shufd(_mm_shuffle_ps(m.m[3], m.m[2], 0x00), 0x80),
											  _mm_shuffle_ps(m.m[2], m.m[1], 0x55)));
			__m128 v1 = _mm_shufd(_mm_shuffle_ps(m.m[1], m.m[0], 0x00), 0xA8);
			__m128 v2 = _mm_shufd(_mm_shuffle_ps(m.m[1], m.m[0], 0x55), 0xA8);
			__m128 v3 = _mm_shufd(_mm_shuffle_ps(m.m[1], m.m[0], 0xAA), 0xA8);
			__m128 v4 = _mm_shufd(_mm_shuffle_ps(m.m[1], m.m[0], 0xFF), 0xA8);
			__m128 s1 = _mm_set_ps(-0.0f,  0.0f, -0.0f,  0.0f);
			__m128 s2 = _mm_set_ps( 0.0f, -0.0f,  0.0f, -0.0f);
			__m128 i1 = _mm_xor_ps(s1, _mm_add_ps(
									   _mm_sub_ps(_mm_mul_ps(v2, f1),
												  _mm_mul_ps(v3, f2)),
												  _mm_mul_ps(v4, f3)));
			__m128 i2 = _mm_xor_ps(s2, _mm_add_ps(
									   _mm_sub_ps(_mm_mul_ps(v1, f1),
												  _mm_mul_ps(v3, f4)),
												  _mm_mul_ps(v4, f5)));
			__m128 i3 = _mm_xor_ps(s1, _mm_add_ps(
									   _mm_sub_ps(_mm_mul_ps(v1, f2),
												  _mm_mul_ps(v2, f4)),
												  _mm_mul_ps(v4, f6)));
			__m128 i4 = _mm_xor_ps(s2, _mm_add_ps(
									   _mm_sub_ps(_mm_mul_ps(v1, f3),
												  _mm_mul_ps(v2, f5)),
												  _mm_mul_ps(v3, f6)));
			__m128 d = _mm_mul_ps(m.m[0], _mm_movelh_ps(_mm_unpacklo_ps(i1, i2),
													    _mm_unpacklo_ps(i3, i4)));
			d = _mm_add_ps(d, _mm_shufd(d, 0x4E));
			d = _mm_add_ps(d, _mm_shufd(d, 0x11));
			d = _mm_div_ps(_mm_set1_ps(1.0f), d);
			return mat4(vec4(_mm_mul_ps(i1, d)),
						vec4(_mm_mul_ps(i2, d)),
						vec4(_mm_mul_ps(i3, d)),
						vec4(_mm_mul_ps(i4, d)));
		}

		// ----------------------------------------------------------------- //

	private:
			// SSE constructor
		inline mat4(const __m128 &_m1, const __m128 &_m2,
		            const __m128 &_m3, const __m128 &_m4) {
			m1 = _m1;
			m2 = _m2;
			m3 = _m3;
			m4 = _m4;
		}

		union {
			__m128 m[4];
			struct {
				__m128 m1;
				__m128 m2;
				__m128 m3;
				__m128 m4;
			};

/*			// This code is waiting for unrestricted unions feature in c++0x
			vec4 v[4];
			struct {
				vec4 v1;
				vec4 v2;
				vec4 v3;
				vec4 v4;
			};
*/
		};

		// Avoid pollution
	#undef _mm_shufd
};

#endif
