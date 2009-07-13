#include "vec4.h"

#ifndef __MAT4_H__
#define __MAT4_H__

typedef union mat4
{
	public:
			// Identity matrix
		inline mat4() {
			m1 = _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f);
			m2 = _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f);
			m3 = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
			m4 = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
		}
			
			// 4 vectors constructor
		inline mat4(const vec4 &_v1, const vec4 &_v2,
		            const vec4 &_v3, const vec4 &_v4) {
			m1 = _v1.m;
			m2 = _v2.m;
			m3 = _v3.m;
			m4 = _v4.m;
		}
			
			// Full float constructor
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

			// Write direct access operator
		inline vec4& operator[](int index) {
			return ((vec4*)this)[index];
		}
		
			// Read direct access operator
		inline const vec4& operator[](int index) const {
			return ((const vec4*)this)[index];
		}

			// Cast operator
		inline operator float*() {
			return (float*)this;
		}
		
			// Const cast operator
		inline operator const float*() const {
			return (const float*)this;
		}

			
		// ----------------------------------------------------------------- //

		inline const mat4 operator *= (float f) {
			m1 = _mm_mul_ps(m1, _mm_set1_ps(f));
			m2 = _mm_mul_ps(m2, _mm_set1_ps(f));
			m3 = _mm_mul_ps(m3, _mm_set1_ps(f));
			m4 = _mm_mul_ps(m4, _mm_set1_ps(f));
				
			return *this;
		}

		inline const mat4 operator * (float f) {
			return (mat4(*this) * f);
		}
		
		inline const vec4 operator * (const vec4 &v) const {
			__m128 m;
			
			m = _mm_add_ps(m, _mm_mul_ps(m1, _mm_shuffle_ps(v.m, v.m, 0x00)));
			m = _mm_add_ps(m, _mm_mul_ps(m2, _mm_shuffle_ps(v.m, v.m, 0x55)));
			m = _mm_add_ps(m, _mm_mul_ps(m3, _mm_shuffle_ps(v.m, v.m, 0xAA)));
			m = _mm_add_ps(m, _mm_mul_ps(m4, _mm_shuffle_ps(v.m, v.m, 0xFF)));
				
			return m;
		}
			
		inline const mat4 operator * (const mat4 &m) const {
			return (mat4(*this) *= m);
		}
			
		inline mat4 operator *= (const mat4 &mx) {
			m1 = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(m1, m1, 0x00), mx.m1), _mm_mul_ps(_mm_shuffle_ps(m1, m1, 0x55), mx.m2)), _mm_mul_ps(_mm_shuffle_ps(m1, m1, 0xAA), mx.m3)), _mm_mul_ps(_mm_shuffle_ps(m1, m1, 0xFF), mx.m4));
			m2 = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(m2, m2, 0x00), mx.m1), _mm_mul_ps(_mm_shuffle_ps(m2, m2, 0x55), mx.m2)), _mm_mul_ps(_mm_shuffle_ps(m2, m2, 0xAA), mx.m3)), _mm_mul_ps(_mm_shuffle_ps(m2, m2, 0xFF), mx.m4));
			m3 = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(m3, m3, 0x00), mx.m1), _mm_mul_ps(_mm_shuffle_ps(m3, m3, 0x55), mx.m2)), _mm_mul_ps(_mm_shuffle_ps(m3, m3, 0xAA), mx.m3)), _mm_mul_ps(_mm_shuffle_ps(m3, m3, 0xFF), mx.m4));
			m4 = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(m4, m4, 0x00), mx.m1), _mm_mul_ps(_mm_shuffle_ps(m4, m4, 0x55), mx.m2)), _mm_mul_ps(_mm_shuffle_ps(m4, m4, 0xAA), mx.m3)), _mm_mul_ps(_mm_shuffle_ps(m4, m4, 0xFF), mx.m4));
				
			return *this;
		}

		// ----------------------------------------------------------------- //
			
		inline mat4 transpose()
		{
			_MM_TRANSPOSE4_PS(m1, m2, m3, m4);
				
			return *this;
		}
			
		inline mat4 trasposed() const
		{
			return mat4(*this).transpose();
		}
			
			// Code was taken and case-optimized from "Streaming SIMD Extension - Inverse of 4x4 Matrix" by Intel
			// Which could be found here: ftp://download.intel.com/design/PentiumIII/sml/24504301.pdf
		inline mat4 invert()
		{
			__m128 row0, row1, row2, row3;
			__m128 det, tmp;
				
			tmp = _mm_unpacklo_ps(m1, m2);
			tmp = _mm_shuffle_ps(tmp, tmp, 0xD8);
				
			row1 = _mm_unpacklo_ps(m3, m4);
			row1 = _mm_shuffle_ps(row1, row1, 0xD8);
				
			row0 = _mm_shuffle_ps(tmp, row1, 0x88);
			row1 = _mm_shuffle_ps(row1, tmp, 0xDD);
				
			tmp = _mm_unpackhi_ps(m1, m2);
			tmp = _mm_shuffle_ps(tmp, tmp, 0xD8);
				
			row3 = _mm_unpackhi_ps(m3, m4);
			row3 = _mm_shuffle_ps(row3, row3, 0xD8);
				
			row2 = _mm_shuffle_ps(tmp, row3, 0x88);
			row3 = _mm_shuffle_ps(row3, tmp, 0xDD);
				
			tmp = _mm_mul_ps(row2, row3);
			tmp = _mm_shuffle_ps(tmp, tmp, 0xB1);
				
			m1 = _mm_mul_ps(row1, tmp);
			m2 = _mm_mul_ps(row0, tmp);
				
			tmp = _mm_shuffle_ps(tmp, tmp, 0x4E);
				
			m1 = _mm_sub_ps(_mm_mul_ps(row1, tmp), m1);
			m2 = _mm_sub_ps(_mm_mul_ps(row0, tmp), m2);
			m2 = _mm_shuffle_ps(m2, m2, 0x4E);
				
			tmp = _mm_mul_ps(row1, row2);
			tmp = _mm_shuffle_ps(tmp, tmp, 0xB1);
				
			m1 = _mm_add_ps(_mm_mul_ps(row3, tmp), m1);
			m4 = _mm_mul_ps(row0, tmp);
				
			tmp = _mm_shuffle_ps(tmp, tmp, 0x4E);
				
			m1 = _mm_sub_ps(m1, _mm_mul_ps(row3, tmp));
			m4 = _mm_sub_ps(_mm_mul_ps(row0, tmp), m4);
			m4 = _mm_shuffle_ps(m4, m4, 0x4E);
				
			tmp = _mm_mul_ps(_mm_shuffle_ps(row1, row1, 0x4E), row3);
			tmp = _mm_shuffle_ps(tmp, tmp, 0xB1);
			row2 = _mm_shuffle_ps(row2, row2, 0x4E);
				
			m1 = _mm_add_ps(_mm_mul_ps(row2, tmp), m1);
			m3 = _mm_mul_ps(row0, tmp);
				
			tmp = _mm_shuffle_ps(tmp, tmp, 0x4E);
				
			m1 = _mm_sub_ps(m1, _mm_mul_ps(row2, tmp));
			m3 = _mm_sub_ps(_mm_mul_ps(row0, tmp), m3);
			m3 = _mm_shuffle_ps(m3, m3, 0x4E);
				
			tmp = _mm_mul_ps(row0, row1);
			tmp = _mm_shuffle_ps(tmp, tmp, 0xB1);
				
			m3 = _mm_add_ps(_mm_mul_ps(row3, tmp), m3);
			m4 = _mm_sub_ps(_mm_mul_ps(row2, tmp), m4);
				
			tmp = _mm_shuffle_ps(tmp, tmp, 0x4E);
				
			m3 = _mm_sub_ps(_mm_mul_ps(row3, tmp), m3);
			m4 = _mm_sub_ps(m4, _mm_mul_ps(row2, tmp));
				
			tmp = _mm_mul_ps(row0, row3);
			tmp = _mm_shuffle_ps(tmp, tmp, 0xB1);
				
			m2 = _mm_sub_ps(m2, _mm_mul_ps(row2, tmp));
			m3 = _mm_add_ps(_mm_mul_ps(row1, tmp), m3);
				
			tmp = _mm_shuffle_ps(tmp, tmp, 0x4E);
				
			m2 = _mm_add_ps(_mm_mul_ps(row2, tmp), m2);
			m3 = _mm_sub_ps(m3, _mm_mul_ps(row1, tmp));
				
			tmp = _mm_mul_ps(row0, row2);
			tmp = _mm_shuffle_ps(tmp, tmp, 0xB1);
				
			m2 = _mm_add_ps(_mm_mul_ps(row3, tmp), m2);
			m4 = _mm_sub_ps(m4, _mm_mul_ps(row1, tmp));
				
			tmp = _mm_shuffle_ps(tmp, tmp, 0x4E);
				
			m2 = _mm_sub_ps(m2, _mm_mul_ps(row3, tmp));
			m4 = _mm_add_ps(_mm_mul_ps(row1, tmp), m4);
				
			det = _mm_mul_ps(row0, m1);
			det = _mm_add_ps(_mm_shuffle_ps(det, det, 0x4E), det);
			det = _mm_add_ss(_mm_shuffle_ps(det, det, 0xB1), det);
			tmp = _mm_rcp_ss(det);
				
			det = _mm_sub_ss(_mm_add_ss(tmp, tmp), _mm_mul_ss(det, _mm_mul_ss(tmp, tmp)));
			det = _mm_shuffle_ps(det, det, 0x00);
				
			m1 = _mm_mul_ps(det, m1);
			m2 = _mm_mul_ps(det, m2);
			m3 = _mm_mul_ps(det, m3);
			m4 = _mm_mul_ps(det, m4);
			
			return *this;
		}
			
		inline const mat4 inverse() const
		{
			return mat4(*this).invert();
		}
			
		// ----------------------------------------------------------------- //
			
	private:
			// SSE constructor
		inline mat4(const __m128 &_m1, const __m128 &_m2,
		            const __m128 &_m3, const __m128 &_m4) {
			m1 = _m1;
			m1 = _m2;
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
		};
			
/*			// This code is waiting for unrestricted unions feature in c++0x
		union {
			vec4 v[4];
			struct {
				vec4 v1;
				vec4 v2;
				vec4 v3;
				vec4 v4;
			};
		};
*/
} mat4;

#endif
