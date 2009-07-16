#ifndef __VEC4_H__
#define __VEC4_H__

#include <stdio.h>
#include <math.h>
#include <xmmintrin.h>

typedef union vec4
{	
		// FIXME until c++0x comes out for mat4 unrestricted unions
	friend union mat4;
		
	private:
			// SSE register
		__m128 m;

			// SSE compatible constructor
		inline vec4(const __m128 &_m) {
			m = _m;
		}

		template <unsigned char target, unsigned mask>
		struct _mask_merger
		{
			enum
			{
				ROW0 = ((target >> (((mask >> (0 * 2)) & 3) << 1)) & 3) << (0 * 2),
				ROW1 = ((target >> (((mask >> (1 * 2)) & 3) << 1)) & 3) << (1 * 2),
				ROW2 = ((target >> (((mask >> (2 * 2)) & 3) << 1)) & 3) << (2 * 2),
				ROW3 = ((target >> (((mask >> (3 * 2)) & 3) << 1)) & 3) << (3 * 2),

				MASK = ROW0 | ROW1 | ROW2 | ROW3,
			};

			private:
				_mask_merger();
		};

		template <unsigned char mask>
		struct _mask_reverser
		{
			enum
			{
				ROW0 = 0 << (((mask >> 0) & 3) * 2),
				ROW1 = 1 << (((mask >> 2) & 3) * 2),
				ROW2 = 2 << (((mask >> 4) & 3) * 2),
				ROW3 = 3 << (((mask >> 6) & 3) * 2),

				MASK = ROW0 | ROW1 | ROW2 | ROW3,
			};

			private:
				_mask_reverser();
		};

			// Swizzle helper
		template <unsigned mask>
		struct _swzl
		{
			friend union vec4;

			private:
					// Refrence to unswizzled self
				__m128 &m;

			public:
				inline _swzl(__m128 &m):m(m) { 
					// Empty
				}

				inline operator const vec4 () {
					return _mm_shuffle_ps(m, m, mask);
				}

					// Swizzle from vec4
				inline _swzl& operator = (const vec4 &v) {
					m = _mm_shuffle_ps(v.m, v.m, _mask_reverser<mask>::MASK);
					return *this;
				}
				 
					// Swizzle from same mask (v1.xyzw = v2.xyzw)
				inline _swzl& operator = (const _swzl &s) {
					m = s.m;
					return *this;
				}

					// Swizzle mask => other_mask (v1.zwxy = v2.xyxy)
				template<unsigned other_mask>
				inline _swzl& operator = (const _swzl<other_mask> &s) {
						// Needed because shuffle below is a macro and is confused by the template commas.
					typedef _mask_merger<other_mask, _mask_reverser<mask>::MASK> merged;
					m = _mm_shuffle_ps(s.m, s.m, merged::MASK);
					return *this;
				}

					// Swizzle of the swizzle, read only (v.xxxx.yyyy)
				template<unsigned other_mask>
				inline const vec4 shuffle_ro() const {
						// Needed because shuffle below is a macro and is confused by the template commas.
					typedef _mask_merger<mask, other_mask> merged;
					return _mm_shuffle_ps(m, m, merged::MASK);
				}

					// Swizzle of the swizzle, read/write (v1.zyxw.wzyx = ...)
				template<unsigned other_mask>
				inline _swzl<_mask_merger<mask, other_mask>::MASK> shuffle_rw() {
					return _swzl<_mask_merger<mask, other_mask>::MASK>(m);
				}

					// Swizzle of the swizzle, read/write const correct
				template<unsigned other_mask>
				inline _swzl<other_mask> shuffle_rw() const {
						// Needed because shuffle below is a macro and is confused by the template commas.
					typedef _mask_merger<mask, other_mask> merged;
					return _mm_shuffle_ps(m, m, merged::MASK);
				}
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

		// ----------------------------------------------------------------- //

			// Swizzled constructor
		template<unsigned mask>
		inline vec4(const _swzl<mask> &v) {
			m = _mm_shuffle_ps(v.m, v.m, mask);
		}

			// Read-write swizzle
		template<unsigned mask>
		inline _swzl<mask> shuffle_rw() {
			return _swzl<mask>(m);
		}

			// Read-write (actual read only) swizzle, const
		template<unsigned mask>
		inline const vec4 shuffle_rw() const {
			return _mm_shuffle_ps(m, m, mask);
		}

			// Read-only swizzle
		template<unsigned mask>
		inline const vec4 shuffle_ro() {
			return _mm_shuffle_ps(m, m, mask);
		}

			// Read-only swizzle, const
		template<unsigned mask>
		inline const vec4 shuffle_ro() const {
			return _mm_shuffle_ps(m, m, mask);
		}

		// ----------------------------------------------------------------- //

			// Write direct access operator
		inline float& operator[](int index) {
			return ((float*)this)[index];
		}
		
			// Read direct access operator
		inline const float& operator[](int index) const {
			return ((const float*)this)[index];
		}

			// Cast operator
		inline operator float* () {
			return (float*)this;
		}
		
			// Const cast operator
		inline operator const float* () const {
			return (const float*)this;
		}

		// ----------------------------------------------------------------- //

		inline vec4 operator += (float f) {
			return (m = _mm_add_ps(m, _mm_set1_ps(f)));
		}
		
		inline vec4 operator += (const vec4 &v) {
			return (m = _mm_add_ps(m, v.m));
		}

		inline vec4 operator -= (float f) {
			return (m = _mm_sub_ps(m, _mm_set1_ps(f)));
		}
		
		inline vec4 operator -= (const vec4 &v) {
			return (m = _mm_sub_ps(m, v.m));
		}

		inline vec4 operator *= (float f) {	
			return (m = _mm_mul_ps(m, _mm_set1_ps(f)));
		}
					
		inline vec4 operator *= (const vec4 &v) {
			return (m = _mm_mul_ps(m, v.m));
		}

		inline vec4 operator /= (float f) {
			return (m = _mm_div_ps(m, _mm_set1_ps(f)));
		}
		
		inline vec4 operator /= (const vec4 &v) {
			return (m = _mm_div_ps(m, v.m));
		}

		// ----------------------------------------------------------------- //

		friend inline const vec4 operator + (const vec4 &, float );
		
		friend inline const vec4 operator + (const vec4 &, const vec4 &);

		inline const vec4 operator - () {
			return _mm_xor_ps(m, _mm_set1_ps(-0.f));
		}

		friend inline const vec4 operator - (const vec4 &);
		
		friend inline const vec4 operator - (const vec4 &, float );

		friend inline const vec4 operator - (const vec4 &, const vec4 &);

		friend inline const vec4 operator * (const vec4 &, float);

		friend inline const vec4 operator * (const vec4 &, const vec4 &);

		friend inline const vec4 operator / (float , const vec4 &);

		friend inline const vec4 operator / (const vec4 &, float);

		friend inline const vec4 operator / (const vec4 &, const vec4 &);

		// ----------------------------------------------------------------- //
		
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
			__m128 t = _mm_or_ps(_mm_and_ps(_mm_set1_ps(-0.f), v.m),
			                     _mm_set1_ps(1 << 23));
			t = _mm_sub_ps(_mm_add_ps(v.m, t), t);
			return _mm_add_ps(t, _mm_and_ps(_mm_cmpgt_ps(v.m, t),
			                                _mm_set1_ps(1.0f)));
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
			__m128 t = _mm_or_ps(_mm_and_ps(_mm_set1_ps(-0.f), v.m),
			                     _mm_set1_ps(1 << 23));
			t = _mm_sub_ps(_mm_add_ps(v.m, t), t);
			return _mm_sub_ps(t, _mm_and_ps(_mm_cmplt_ps(v.m, t),
			                                _mm_set1_ps(1.0f)));
		}

		friend inline const vec4 fract(const vec4 &v) {
			__m128 t = _mm_or_ps(_mm_and_ps(_mm_set1_ps(-0.f), v.m),
			                     _mm_set1_ps(1 << 23));
			return _mm_sub_ps(v.m, _mm_sub_ps(_mm_add_ps(v.m, t), t));
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

		friend inline const vec4 mix(const vec4 &v0, const vec4 &v1, float f) {
			return _mm_sub_ps(v1.m, _mm_mul_ps(_mm_set1_ps(f),
			                                   _mm_add_ps(v0.m, v1.m)));
		}

		friend inline const vec4 mix(const vec4 &v0, const vec4 &v1,
		                             const vec4 &v2) {
			return _mm_sub_ps(v1.m, _mm_mul_ps(v2.m, _mm_add_ps(v0.m, v1.m)));
		}

		friend inline const vec4 mod(const vec4 &v, float f) {
			__m128 r = _mm_div_ps(v.m, _mm_set1_ps(f));
			__m128 t = _mm_or_ps(_mm_and_ps(_mm_set1_ps(-0.f), r),
			                     _mm_set1_ps(1 << 23));
			return _mm_sub_ps(v.m, _mm_mul_ps(_mm_sub_ps(_mm_add_ps(r, t), t),
			                                   _mm_set1_ps(f)));
		}

		friend inline const vec4 mod(const vec4 &v0, const vec4 &v1) {
			__m128 r = _mm_div_ps(v0.m, v1.m);
			__m128 t = _mm_or_ps(_mm_and_ps(_mm_set1_ps(-0.f), r),
			                     _mm_set1_ps(1 << 23));
			return _mm_sub_ps(v0.m, _mm_mul_ps(_mm_sub_ps(_mm_add_ps(r, t), t),
			                                   v1.m));
		}

		friend inline const vec4 sign(const vec4 &v) {
			return _mm_and_ps(_mm_or_ps(_mm_and_ps(v.m, _mm_set1_ps(-0.f)),
			                            _mm_set1_ps(1.0f)),
			                  _mm_cmpneq_ps(v.m, _mm_setzero_ps()));
		}

		friend inline const vec4 smoothstep(float f1, float f2,
		                                    const vec4 &v) {
			__m128 v0 = _mm_set1_ps(f1);
			__m128 c = _mm_max_ps(_mm_setzero_ps(),
			                      _mm_min_ps(_mm_set1_ps(1.0f),
			                                 _mm_sub_ps(_mm_sub_ps(v.m, v0),
											 _mm_sub_ps(_mm_set1_ps(f2), v0))));
			return _mm_mul_ps(_mm_mul_ps(c, c),
			                  _mm_sub_ps(_mm_set1_ps(3.0f),
			                             _mm_mul_ps(_mm_set1_ps(2.0f), c)));
		}
		
		friend inline const vec4 smoothstep(const vec4 &v0,
		                                    const vec4 &v1, const vec4 &v2) {
			__m128 c = _mm_max_ps(_mm_setzero_ps(),
			                      _mm_min_ps(_mm_set1_ps(1.0f),
			                                 _mm_sub_ps(_mm_sub_ps(v2.m, v0.m),
			                                            _mm_sub_ps(v1.m, v0.m))));
			return _mm_mul_ps(_mm_mul_ps(c, c),
			                  _mm_sub_ps(_mm_set1_ps(3.0f),
			                             _mm_mul_ps(_mm_set1_ps(2.0f), c)));
		}

		friend inline const vec4 step(float f, const vec4 &v) {
			return _mm_and_ps(_mm_cmple_ps(v.m, _mm_set1_ps(f)),
			                  _mm_set1_ps(1.0f));
		}

		friend inline const vec4 step(const vec4 &v0, const vec4 &v1) {
			return _mm_and_ps(_mm_cmple_ps(v0.m, v1.m), _mm_set1_ps(1.0f));
		}

		// ----------------------------------------------------------------- //

		friend inline float distance(const vec4 &v0, const vec4 &v1) {
			__m128 l = _mm_sub_ps(v0.m, v1.m);
			l = _mm_mul_ps(l, l);
			l = _mm_add_ps(l, _mm_shuffle_ps(l, l, 0x4E));
			return _mm_cvtss_f32(_mm_sqrt_ss(_mm_add_ss(l,
			                                 _mm_shuffle_ps(l, l, 0x11))));
		}

		friend inline float dot(const vec4 &v0, const vec4 &v1) {
			__m128 l = _mm_mul_ps(v0.m, v1.m);
			l = _mm_add_ps(l, _mm_shuffle_ps(l, l, 0x4E));
			return _mm_cvtss_f32(_mm_add_ss(l, _mm_shuffle_ps(l, l, 0x11)));

		}

		friend inline const vec4 faceforward(const vec4 &v0,
		                                     const vec4 &v1, const vec4 &v2) {
			__m128 l = _mm_mul_ps(v2.m, v1.m);
			l = _mm_add_ps(l, _mm_shuffle_ps(l, l, 0x4E));
			return _mm_xor_ps(_mm_and_ps(_mm_cmpnlt_ps(
			        _mm_add_ps(l, _mm_shuffle_ps(l, l, 0x11)),
			        _mm_setzero_ps()), _mm_set1_ps(-0.f)), v0.m);
		}

		friend inline float length(const vec4 &v) {
			__m128 l = _mm_mul_ps(v.m, v.m);
			l = _mm_add_ps(l, _mm_shuffle_ps(l, l, 0x4E));
			return _mm_cvtss_f32(_mm_sqrt_ss(_mm_add_ss(l,
			                                 _mm_shuffle_ps(l, l, 0x11))));
		}

		friend inline const vec4 normalize(const vec4 &v) {
			__m128 l = _mm_mul_ps(v.m, v.m);
			l = _mm_add_ps(l, _mm_shuffle_ps(l, l, 0x4E));
			return _mm_div_ps(v.m, _mm_sqrt_ps(_mm_add_ps(l,
			                                   _mm_shuffle_ps(l, l, 0x11))));
		}

		friend inline const vec4 reflect(const vec4 &v0, const vec4 &v1) {
			__m128 l = _mm_mul_ps(v1.m, v0.m);
			l = _mm_add_ps(l, _mm_shuffle_ps(l, l, 0x4E));
			l = _mm_add_ps(l, _mm_shuffle_ps(l, l, 0x11));
			return _mm_sub_ps(v0.m, _mm_mul_ps(_mm_add_ps(l, l), v1.m));
		}

		friend inline const vec4 refract(const vec4 &v0, const vec4 &v1, float f) {
			__m128 d = _mm_mul_ps(v1.m, v0.m);
			d = _mm_add_ps(d, _mm_shuffle_ps(d, d, 0x4E));
			d = _mm_add_ps(d, _mm_shuffle_ps(d, d, 0x11));
			__m128 e = _mm_set1_ps(f);
			__m128 k = _mm_sub_ps(_mm_set1_ps(1.0f),
			                      _mm_mul_ps(_mm_mul_ps(e, e),
			                                 _mm_sub_ps(_mm_set1_ps(1.0f),
			                                            _mm_mul_ps(d, d))));
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
			return (_mm_movemask_ps(_mm_cmpneq_ps(v0.m, v1.m)) == 0xF);
		}

		// ----------------------------------------------------------------- //

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
} vec4;

inline const vec4 operator + (const vec4 &v, float f) {
	return _mm_add_ps(v.m, _mm_set1_ps(f));
}

inline const vec4 operator + (const vec4 &v0, const vec4 &v1) {
	return _mm_add_ps(v0.m, v1.m);
}

inline const vec4 operator - (const vec4 &v) {
	return _mm_sub_ps(_mm_setzero_ps(), v.m);
}

inline const vec4 operator - (const vec4 &v, float f) {
	return _mm_sub_ps(v.m, _mm_set1_ps(f));
}

inline const vec4 operator - (const vec4 &v0, const vec4 &v1) {
	return _mm_sub_ps(v0.m, v1.m);
}

inline const vec4 operator * (const vec4 &v, float f) {
	return _mm_mul_ps(v.m, _mm_set1_ps(f));
}

inline const vec4 operator * (const vec4 &v0, const vec4 &v1) {
	return _mm_mul_ps(v0.m, v1.m);
}

inline const vec4 operator / (float f, const vec4 &v) {
	return _mm_div_ps(_mm_set1_ps(f), v.m);
}

inline const vec4 operator / (const vec4 &v, float f) {
	return _mm_div_ps(v.m, _mm_set1_ps(f));
}

inline const vec4 operator / (const vec4 &v0, const vec4 &v1) {
	return _mm_div_ps(v0.m, v1.m);
}		

#define wzyx shuffle_rw<_MM_SHUFFLE(0,1,2,3)>()
#define zwyx shuffle_rw<_MM_SHUFFLE(0,1,3,2)>()
#define wyzx shuffle_rw<_MM_SHUFFLE(0,2,1,3)>()
#define ywzx shuffle_rw<_MM_SHUFFLE(0,2,3,1)>()
#define zywx shuffle_rw<_MM_SHUFFLE(0,3,1,2)>()
#define yzwx shuffle_rw<_MM_SHUFFLE(0,3,2,1)>()
#define wzxy shuffle_rw<_MM_SHUFFLE(1,0,2,3)>()
#define zwxy shuffle_rw<_MM_SHUFFLE(1,0,3,2)>()
#define wxzy shuffle_rw<_MM_SHUFFLE(1,2,0,3)>()
#define xwzy shuffle_rw<_MM_SHUFFLE(1,2,3,0)>()
#define zxwy shuffle_rw<_MM_SHUFFLE(1,3,0,2)>()
#define xzwy shuffle_rw<_MM_SHUFFLE(1,3,2,0)>()
#define wyxz shuffle_rw<_MM_SHUFFLE(2,0,1,3)>()
#define ywxz shuffle_rw<_MM_SHUFFLE(2,0,3,1)>()
#define wxyz shuffle_rw<_MM_SHUFFLE(2,1,0,3)>()
#define xwyz shuffle_rw<_MM_SHUFFLE(2,1,3,0)>()
#define yxwz shuffle_rw<_MM_SHUFFLE(2,3,0,1)>()
#define xywz shuffle_rw<_MM_SHUFFLE(2,3,1,0)>()
#define zyxw shuffle_rw<_MM_SHUFFLE(3,0,1,2)>()
#define yzxw shuffle_rw<_MM_SHUFFLE(3,0,2,1)>()
#define zxyw shuffle_rw<_MM_SHUFFLE(3,1,0,2)>()
#define xzyw shuffle_rw<_MM_SHUFFLE(3,1,2,0)>()
#define yxzw shuffle_rw<_MM_SHUFFLE(3,2,0,1)>()
#define xyzw shuffle_rw<_MM_SHUFFLE(3,2,1,0)>()

#define xxxx shuffle_ro<_MM_SHUFFLE(0,0,0,0)>()
#define yxxx shuffle_ro<_MM_SHUFFLE(0,0,0,1)>()
#define zxxx shuffle_ro<_MM_SHUFFLE(0,0,0,2)>()
#define wxxx shuffle_ro<_MM_SHUFFLE(0,0,0,3)>()
#define xyxx shuffle_ro<_MM_SHUFFLE(0,0,1,0)>()
#define yyxx shuffle_ro<_MM_SHUFFLE(0,0,1,1)>()
#define zyxx shuffle_ro<_MM_SHUFFLE(0,0,1,2)>()
#define wyxx shuffle_ro<_MM_SHUFFLE(0,0,1,3)>()
#define xzxx shuffle_ro<_MM_SHUFFLE(0,0,2,0)>()
#define yzxx shuffle_ro<_MM_SHUFFLE(0,0,2,1)>()
#define zzxx shuffle_ro<_MM_SHUFFLE(0,0,2,2)>()
#define wzxx shuffle_ro<_MM_SHUFFLE(0,0,2,3)>()
#define xwxx shuffle_ro<_MM_SHUFFLE(0,0,3,0)>()
#define ywxx shuffle_ro<_MM_SHUFFLE(0,0,3,1)>()
#define zwxx shuffle_ro<_MM_SHUFFLE(0,0,3,2)>()
#define wwxx shuffle_ro<_MM_SHUFFLE(0,0,3,3)>()
#define xxyx shuffle_ro<_MM_SHUFFLE(0,1,0,0)>()
#define yxyx shuffle_ro<_MM_SHUFFLE(0,1,0,1)>()
#define zxyx shuffle_ro<_MM_SHUFFLE(0,1,0,2)>()
#define wxyx shuffle_ro<_MM_SHUFFLE(0,1,0,3)>()
#define xyyx shuffle_ro<_MM_SHUFFLE(0,1,1,0)>()
#define yyyx shuffle_ro<_MM_SHUFFLE(0,1,1,1)>()
#define zyyx shuffle_ro<_MM_SHUFFLE(0,1,1,2)>()
#define wyyx shuffle_ro<_MM_SHUFFLE(0,1,1,3)>()
#define xzyx shuffle_ro<_MM_SHUFFLE(0,1,2,0)>()
#define yzyx shuffle_ro<_MM_SHUFFLE(0,1,2,1)>()
#define zzyx shuffle_ro<_MM_SHUFFLE(0,1,2,2)>()
#define xwyx shuffle_ro<_MM_SHUFFLE(0,1,3,0)>()
#define ywyx shuffle_ro<_MM_SHUFFLE(0,1,3,1)>()
#define wwyx shuffle_ro<_MM_SHUFFLE(0,1,3,3)>()
#define xxzx shuffle_ro<_MM_SHUFFLE(0,2,0,0)>()
#define yxzx shuffle_ro<_MM_SHUFFLE(0,2,0,1)>()
#define zxzx shuffle_ro<_MM_SHUFFLE(0,2,0,2)>()
#define wxzx shuffle_ro<_MM_SHUFFLE(0,2,0,3)>()
#define xyzx shuffle_ro<_MM_SHUFFLE(0,2,1,0)>()
#define yyzx shuffle_ro<_MM_SHUFFLE(0,2,1,1)>()
#define zyzx shuffle_ro<_MM_SHUFFLE(0,2,1,2)>()
#define xzzx shuffle_ro<_MM_SHUFFLE(0,2,2,0)>()
#define yzzx shuffle_ro<_MM_SHUFFLE(0,2,2,1)>()
#define zzzx shuffle_ro<_MM_SHUFFLE(0,2,2,2)>()
#define wzzx shuffle_ro<_MM_SHUFFLE(0,2,2,3)>()
#define xwzx shuffle_ro<_MM_SHUFFLE(0,2,3,0)>()
#define zwzx shuffle_ro<_MM_SHUFFLE(0,2,3,2)>()
#define wwzx shuffle_ro<_MM_SHUFFLE(0,2,3,3)>()
#define xxwx shuffle_ro<_MM_SHUFFLE(0,3,0,0)>()
#define yxwx shuffle_ro<_MM_SHUFFLE(0,3,0,1)>()
#define zxwx shuffle_ro<_MM_SHUFFLE(0,3,0,2)>()
#define wxwx shuffle_ro<_MM_SHUFFLE(0,3,0,3)>()
#define xywx shuffle_ro<_MM_SHUFFLE(0,3,1,0)>()
#define yywx shuffle_ro<_MM_SHUFFLE(0,3,1,1)>()
#define wywx shuffle_ro<_MM_SHUFFLE(0,3,1,3)>()
#define xzwx shuffle_ro<_MM_SHUFFLE(0,3,2,0)>()
#define zzwx shuffle_ro<_MM_SHUFFLE(0,3,2,2)>()
#define wzwx shuffle_ro<_MM_SHUFFLE(0,3,2,3)>()
#define xwwx shuffle_ro<_MM_SHUFFLE(0,3,3,0)>()
#define ywwx shuffle_ro<_MM_SHUFFLE(0,3,3,1)>()
#define zwwx shuffle_ro<_MM_SHUFFLE(0,3,3,2)>()
#define wwwx shuffle_ro<_MM_SHUFFLE(0,3,3,3)>()
#define xxxy shuffle_ro<_MM_SHUFFLE(1,0,0,0)>()
#define yxxy shuffle_ro<_MM_SHUFFLE(1,0,0,1)>()
#define zxxy shuffle_ro<_MM_SHUFFLE(1,0,0,2)>()
#define wxxy shuffle_ro<_MM_SHUFFLE(1,0,0,3)>()
#define xyxy shuffle_ro<_MM_SHUFFLE(1,0,1,0)>()
#define yyxy shuffle_ro<_MM_SHUFFLE(1,0,1,1)>()
#define zyxy shuffle_ro<_MM_SHUFFLE(1,0,1,2)>()
#define wyxy shuffle_ro<_MM_SHUFFLE(1,0,1,3)>()
#define xzxy shuffle_ro<_MM_SHUFFLE(1,0,2,0)>()
#define yzxy shuffle_ro<_MM_SHUFFLE(1,0,2,1)>()
#define zzxy shuffle_ro<_MM_SHUFFLE(1,0,2,2)>()
#define xwxy shuffle_ro<_MM_SHUFFLE(1,0,3,0)>()
#define ywxy shuffle_ro<_MM_SHUFFLE(1,0,3,1)>()
#define wwxy shuffle_ro<_MM_SHUFFLE(1,0,3,3)>()
#define xxyy shuffle_ro<_MM_SHUFFLE(1,1,0,0)>()
#define yxyy shuffle_ro<_MM_SHUFFLE(1,1,0,1)>()
#define zxyy shuffle_ro<_MM_SHUFFLE(1,1,0,2)>()
#define wxyy shuffle_ro<_MM_SHUFFLE(1,1,0,3)>()
#define xyyy shuffle_ro<_MM_SHUFFLE(1,1,1,0)>()
#define yyyy shuffle_ro<_MM_SHUFFLE(1,1,1,1)>()
#define zyyy shuffle_ro<_MM_SHUFFLE(1,1,1,2)>()
#define wyyy shuffle_ro<_MM_SHUFFLE(1,1,1,3)>()
#define xzyy shuffle_ro<_MM_SHUFFLE(1,1,2,0)>()
#define yzyy shuffle_ro<_MM_SHUFFLE(1,1,2,1)>()
#define zzyy shuffle_ro<_MM_SHUFFLE(1,1,2,2)>()
#define wzyy shuffle_ro<_MM_SHUFFLE(1,1,2,3)>()
#define xwyy shuffle_ro<_MM_SHUFFLE(1,1,3,0)>()
#define ywyy shuffle_ro<_MM_SHUFFLE(1,1,3,1)>()
#define zwyy shuffle_ro<_MM_SHUFFLE(1,1,3,2)>()
#define wwyy shuffle_ro<_MM_SHUFFLE(1,1,3,3)>()
#define xxzy shuffle_ro<_MM_SHUFFLE(1,2,0,0)>()
#define yxzy shuffle_ro<_MM_SHUFFLE(1,2,0,1)>()
#define zxzy shuffle_ro<_MM_SHUFFLE(1,2,0,2)>()
#define xyzy shuffle_ro<_MM_SHUFFLE(1,2,1,0)>()
#define yyzy shuffle_ro<_MM_SHUFFLE(1,2,1,1)>()
#define zyzy shuffle_ro<_MM_SHUFFLE(1,2,1,2)>()
#define wyzy shuffle_ro<_MM_SHUFFLE(1,2,1,3)>()
#define xzzy shuffle_ro<_MM_SHUFFLE(1,2,2,0)>()
#define yzzy shuffle_ro<_MM_SHUFFLE(1,2,2,1)>()
#define zzzy shuffle_ro<_MM_SHUFFLE(1,2,2,2)>()
#define wzzy shuffle_ro<_MM_SHUFFLE(1,2,2,3)>()
#define ywzy shuffle_ro<_MM_SHUFFLE(1,2,3,1)>()
#define zwzy shuffle_ro<_MM_SHUFFLE(1,2,3,2)>()
#define wwzy shuffle_ro<_MM_SHUFFLE(1,2,3,3)>()
#define xxwy shuffle_ro<_MM_SHUFFLE(1,3,0,0)>()
#define yxwy shuffle_ro<_MM_SHUFFLE(1,3,0,1)>()
#define wxwy shuffle_ro<_MM_SHUFFLE(1,3,0,3)>()
#define xywy shuffle_ro<_MM_SHUFFLE(1,3,1,0)>()
#define yywy shuffle_ro<_MM_SHUFFLE(1,3,1,1)>()
#define zywy shuffle_ro<_MM_SHUFFLE(1,3,1,2)>()
#define wywy shuffle_ro<_MM_SHUFFLE(1,3,1,3)>()
#define yzwy shuffle_ro<_MM_SHUFFLE(1,3,2,1)>()
#define zzwy shuffle_ro<_MM_SHUFFLE(1,3,2,2)>()
#define wzwy shuffle_ro<_MM_SHUFFLE(1,3,2,3)>()
#define xwwy shuffle_ro<_MM_SHUFFLE(1,3,3,0)>()
#define ywwy shuffle_ro<_MM_SHUFFLE(1,3,3,1)>()
#define zwwy shuffle_ro<_MM_SHUFFLE(1,3,3,2)>()
#define wwwy shuffle_ro<_MM_SHUFFLE(1,3,3,3)>()
#define xxxz shuffle_ro<_MM_SHUFFLE(2,0,0,0)>()
#define yxxz shuffle_ro<_MM_SHUFFLE(2,0,0,1)>()
#define zxxz shuffle_ro<_MM_SHUFFLE(2,0,0,2)>()
#define wxxz shuffle_ro<_MM_SHUFFLE(2,0,0,3)>()
#define xyxz shuffle_ro<_MM_SHUFFLE(2,0,1,0)>()
#define yyxz shuffle_ro<_MM_SHUFFLE(2,0,1,1)>()
#define zyxz shuffle_ro<_MM_SHUFFLE(2,0,1,2)>()
#define xzxz shuffle_ro<_MM_SHUFFLE(2,0,2,0)>()
#define yzxz shuffle_ro<_MM_SHUFFLE(2,0,2,1)>()
#define zzxz shuffle_ro<_MM_SHUFFLE(2,0,2,2)>()
#define wzxz shuffle_ro<_MM_SHUFFLE(2,0,2,3)>()
#define xwxz shuffle_ro<_MM_SHUFFLE(2,0,3,0)>()
#define zwxz shuffle_ro<_MM_SHUFFLE(2,0,3,2)>()
#define wwxz shuffle_ro<_MM_SHUFFLE(2,0,3,3)>()
#define xxyz shuffle_ro<_MM_SHUFFLE(2,1,0,0)>()
#define yxyz shuffle_ro<_MM_SHUFFLE(2,1,0,1)>()
#define zxyz shuffle_ro<_MM_SHUFFLE(2,1,0,2)>()
#define xyyz shuffle_ro<_MM_SHUFFLE(2,1,1,0)>()
#define yyyz shuffle_ro<_MM_SHUFFLE(2,1,1,1)>()
#define zyyz shuffle_ro<_MM_SHUFFLE(2,1,1,2)>()
#define wyyz shuffle_ro<_MM_SHUFFLE(2,1,1,3)>()
#define xzyz shuffle_ro<_MM_SHUFFLE(2,1,2,0)>()
#define yzyz shuffle_ro<_MM_SHUFFLE(2,1,2,1)>()
#define zzyz shuffle_ro<_MM_SHUFFLE(2,1,2,2)>()
#define wzyz shuffle_ro<_MM_SHUFFLE(2,1,2,3)>()
#define ywyz shuffle_ro<_MM_SHUFFLE(2,1,3,1)>()
#define zwyz shuffle_ro<_MM_SHUFFLE(2,1,3,2)>()
#define wwyz shuffle_ro<_MM_SHUFFLE(2,1,3,3)>()
#define xxzz shuffle_ro<_MM_SHUFFLE(2,2,0,0)>()
#define yxzz shuffle_ro<_MM_SHUFFLE(2,2,0,1)>()
#define zxzz shuffle_ro<_MM_SHUFFLE(2,2,0,2)>()
#define wxzz shuffle_ro<_MM_SHUFFLE(2,2,0,3)>()
#define xyzz shuffle_ro<_MM_SHUFFLE(2,2,1,0)>()
#define yyzz shuffle_ro<_MM_SHUFFLE(2,2,1,1)>()
#define zyzz shuffle_ro<_MM_SHUFFLE(2,2,1,2)>()
#define wyzz shuffle_ro<_MM_SHUFFLE(2,2,1,3)>()
#define xzzz shuffle_ro<_MM_SHUFFLE(2,2,2,0)>()
#define yzzz shuffle_ro<_MM_SHUFFLE(2,2,2,1)>()
#define zzzz shuffle_ro<_MM_SHUFFLE(2,2,2,2)>()
#define wzzz shuffle_ro<_MM_SHUFFLE(2,2,2,3)>()
#define xwzz shuffle_ro<_MM_SHUFFLE(2,2,3,0)>()
#define ywzz shuffle_ro<_MM_SHUFFLE(2,2,3,1)>()
#define zwzz shuffle_ro<_MM_SHUFFLE(2,2,3,2)>()
#define wwzz shuffle_ro<_MM_SHUFFLE(2,2,3,3)>()
#define xxwz shuffle_ro<_MM_SHUFFLE(2,3,0,0)>()
#define zxwz shuffle_ro<_MM_SHUFFLE(2,3,0,2)>()
#define wxwz shuffle_ro<_MM_SHUFFLE(2,3,0,3)>()
#define yywz shuffle_ro<_MM_SHUFFLE(2,3,1,1)>()
#define zywz shuffle_ro<_MM_SHUFFLE(2,3,1,2)>()
#define wywz shuffle_ro<_MM_SHUFFLE(2,3,1,3)>()
#define xzwz shuffle_ro<_MM_SHUFFLE(2,3,2,0)>()
#define yzwz shuffle_ro<_MM_SHUFFLE(2,3,2,1)>()
#define zzwz shuffle_ro<_MM_SHUFFLE(2,3,2,2)>()
#define wzwz shuffle_ro<_MM_SHUFFLE(2,3,2,3)>()
#define xwwz shuffle_ro<_MM_SHUFFLE(2,3,3,0)>()
#define ywwz shuffle_ro<_MM_SHUFFLE(2,3,3,1)>()
#define zwwz shuffle_ro<_MM_SHUFFLE(2,3,3,2)>()
#define wwwz shuffle_ro<_MM_SHUFFLE(2,3,3,3)>()
#define xxxw shuffle_ro<_MM_SHUFFLE(3,0,0,0)>()
#define yxxw shuffle_ro<_MM_SHUFFLE(3,0,0,1)>()
#define zxxw shuffle_ro<_MM_SHUFFLE(3,0,0,2)>()
#define wxxw shuffle_ro<_MM_SHUFFLE(3,0,0,3)>()
#define xyxw shuffle_ro<_MM_SHUFFLE(3,0,1,0)>()
#define yyxw shuffle_ro<_MM_SHUFFLE(3,0,1,1)>()
#define wyxw shuffle_ro<_MM_SHUFFLE(3,0,1,3)>()
#define xzxw shuffle_ro<_MM_SHUFFLE(3,0,2,0)>()
#define zzxw shuffle_ro<_MM_SHUFFLE(3,0,2,2)>()
#define wzxw shuffle_ro<_MM_SHUFFLE(3,0,2,3)>()
#define xwxw shuffle_ro<_MM_SHUFFLE(3,0,3,0)>()
#define ywxw shuffle_ro<_MM_SHUFFLE(3,0,3,1)>()
#define zwxw shuffle_ro<_MM_SHUFFLE(3,0,3,2)>()
#define wwxw shuffle_ro<_MM_SHUFFLE(3,0,3,3)>()
#define xxyw shuffle_ro<_MM_SHUFFLE(3,1,0,0)>()
#define yxyw shuffle_ro<_MM_SHUFFLE(3,1,0,1)>()
#define wxyw shuffle_ro<_MM_SHUFFLE(3,1,0,3)>()
#define xyyw shuffle_ro<_MM_SHUFFLE(3,1,1,0)>()
#define yyyw shuffle_ro<_MM_SHUFFLE(3,1,1,1)>()
#define zyyw shuffle_ro<_MM_SHUFFLE(3,1,1,2)>()
#define wyyw shuffle_ro<_MM_SHUFFLE(3,1,1,3)>()
#define yzyw shuffle_ro<_MM_SHUFFLE(3,1,2,1)>()
#define zzyw shuffle_ro<_MM_SHUFFLE(3,1,2,2)>()
#define wzyw shuffle_ro<_MM_SHUFFLE(3,1,2,3)>()
#define xwyw shuffle_ro<_MM_SHUFFLE(3,1,3,0)>()
#define ywyw shuffle_ro<_MM_SHUFFLE(3,1,3,1)>()
#define zwyw shuffle_ro<_MM_SHUFFLE(3,1,3,2)>()
#define wwyw shuffle_ro<_MM_SHUFFLE(3,1,3,3)>()
#define xxzw shuffle_ro<_MM_SHUFFLE(3,2,0,0)>()
#define zxzw shuffle_ro<_MM_SHUFFLE(3,2,0,2)>()
#define wxzw shuffle_ro<_MM_SHUFFLE(3,2,0,3)>()
#define yyzw shuffle_ro<_MM_SHUFFLE(3,2,1,1)>()
#define zyzw shuffle_ro<_MM_SHUFFLE(3,2,1,2)>()
#define wyzw shuffle_ro<_MM_SHUFFLE(3,2,1,3)>()
#define xzzw shuffle_ro<_MM_SHUFFLE(3,2,2,0)>()
#define yzzw shuffle_ro<_MM_SHUFFLE(3,2,2,1)>()
#define zzzw shuffle_ro<_MM_SHUFFLE(3,2,2,2)>()
#define wzzw shuffle_ro<_MM_SHUFFLE(3,2,2,3)>()
#define xwzw shuffle_ro<_MM_SHUFFLE(3,2,3,0)>()
#define ywzw shuffle_ro<_MM_SHUFFLE(3,2,3,1)>()
#define zwzw shuffle_ro<_MM_SHUFFLE(3,2,3,2)>()
#define wwzw shuffle_ro<_MM_SHUFFLE(3,2,3,3)>()
#define xxww shuffle_ro<_MM_SHUFFLE(3,3,0,0)>()
#define yxww shuffle_ro<_MM_SHUFFLE(3,3,0,1)>()
#define zxww shuffle_ro<_MM_SHUFFLE(3,3,0,2)>()
#define wxww shuffle_ro<_MM_SHUFFLE(3,3,0,3)>()
#define xyww shuffle_ro<_MM_SHUFFLE(3,3,1,0)>()
#define yyww shuffle_ro<_MM_SHUFFLE(3,3,1,1)>()
#define zyww shuffle_ro<_MM_SHUFFLE(3,3,1,2)>()
#define wyww shuffle_ro<_MM_SHUFFLE(3,3,1,3)>()
#define xzww shuffle_ro<_MM_SHUFFLE(3,3,2,0)>()
#define yzww shuffle_ro<_MM_SHUFFLE(3,3,2,1)>()
#define zzww shuffle_ro<_MM_SHUFFLE(3,3,2,2)>()
#define wzww shuffle_ro<_MM_SHUFFLE(3,3,2,3)>()
#define xwww shuffle_ro<_MM_SHUFFLE(3,3,3,0)>()
#define ywww shuffle_ro<_MM_SHUFFLE(3,3,3,1)>()
#define zwww shuffle_ro<_MM_SHUFFLE(3,3,3,2)>()
#define wwww shuffle_ro<_MM_SHUFFLE(3,3,3,3)>()

#endif
