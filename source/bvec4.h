#ifndef __BVEC4_H__
#define __BVEC4_H__

#include "vec4.h"

typedef union bvec4
{
	public:
		bvec4(bool _x, bool _y, bool _z, bool _w):
			x(_x), y(_y), z(_z), w(_w) {
			// Empty
		}

		// ----------------------------------------------------------------- //
/*
		 	// not is a reserved word in C++03.
		friend inline const bvec4 not(const vec4 &v0, const vec4 &v1) {
			return bvec4(0xF - all);
		}
*/
		friend inline const bvec4 equal(const vec4 &v0, const vec4 &v1);

		friend inline const bvec4 notEqual(const vec4 &v0, const vec4 &v1);

		friend inline const bvec4 greaterThan(const vec4 &v0, const vec4 &v1);

		friend inline const bvec4 greaterThanEqual(const vec4 &v0, const vec4 &v1);

		friend inline const bvec4 lessThan(const vec4 &v0, const vec4 &v1);

		friend inline const bvec4 lessThanEqual(const vec4 &v0, const vec4 &v1);

		// ----------------------------------------------------------------- //

		friend inline const bvec4 isnan(const vec4 &v);

		friend inline const bvec4 isinf(const vec4 &v);

		// ----------------------------------------------------------------- //

		friend inline bool any(const bvec4 &b);
		
		friend inline bool all(const bvec4 &b);

		// ----------------------------------------------------------------- //

		friend inline bool operator == (const bvec4 &b0, const bvec4 &b1) {
			return (b0.all & 0xF) == (b1.all & 0xF);
		}

		friend inline bool operator != (const bvec4 &b0, const bvec4 &b1) {
			return (b0.all & 0xF) != (b1.all & 0xF);
		}
		
		// ----------------------------------------------------------------- //

			// Vertex / Vector 
		struct {
			bool x :1;
			bool y :1;
			bool z :1;
			bool w :1;
		};
			// Color
		struct {
			bool r :1;
			bool g :1;
			bool b :1;
			bool a :1;
		};
			// Texture coordinates
		struct {
			bool s :1;
			bool t :1;
			bool p :1;
			bool q :1;
		};

	private:

			// Mask created from _mm_movemask_ps 
		explicit bvec4(int mask) {
			all = mask;
		};

		unsigned char all;
} bvec4;

inline const bvec4 equal(const vec4 &v0, const vec4 &v1) {
	return bvec4(_mm_movemask_ps(_mm_cmpeq_ps(v0.m, v1.m)));
}

inline const bvec4 notEqual(const vec4 &v0, const vec4 &v1) {
	return bvec4(_mm_movemask_ps(_mm_cmpneq_ps(v0.m, v1.m)));
}

inline const bvec4 greaterThan(const vec4 &v0, const vec4 &v1) {
	return bvec4(_mm_movemask_ps(_mm_cmpgt_ps(v0.m, v1.m)));
}

inline const bvec4 greaterThanEqual(const vec4 &v0, const vec4 &v1) {
	return bvec4(_mm_movemask_ps(_mm_cmpge_ps(v0.m, v1.m)));
}

inline const bvec4 lessThan(const vec4 &v0, const vec4 &v1) {
	return bvec4(_mm_movemask_ps(_mm_cmplt_ps(v0.m, v1.m)));
}

inline const bvec4 lessThanEqual(const vec4 &v0, const vec4 &v1) {
	return bvec4(_mm_movemask_ps(_mm_cmple_ps(v0.m, v1.m)));
}

// ------------------------------------------------------------------------- //

inline const bvec4 isnan(const vec4 &v) {
	return bvec4(_mm_movemask_ps(_mm_cmpunord_ps(v.m, v.m)));
}

inline const bvec4 isinf(const vec4 &v) {
	return bvec4(_mm_movemask_ps(_mm_cmpeq_ps(
				 _mm_andnot_ps(_mm_set1_ps(-0.f), v.m),
				 _mm_castsi128_ps(_mm_set1_epi32(0x7F800000)))));
}

// ------------------------------------------------------------------------- //

inline bool any(const bvec4 &b) {
	return (b.all & 0xF) != 0x0;
}

inline bool all(const bvec4 &b) {
	return (b.all & 0xF) == 0xF;
}

#endif
