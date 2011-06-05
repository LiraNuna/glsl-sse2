#ifndef __DVEC4_H__
#define __DVEC4_H__

class dvec4
{
	private:
			// The actual swizzle code, since we operate on two xmm registers
		template <unsigned mask>
		static inline dvec4 shuffle(const dvec4 &v)
		{
			const __m128d &S1 = v.m[(mask >> 1) & 1];
			const __m128d &S2 = v.m[(mask >> 3) & 1];
			const __m128d &S3 = v.m[(mask >> 5) & 1];
			const __m128d &S4 = v.m[(mask >> 7) & 1];

			return dvec4(_mm_shuffle_pd(S1, S2, _mask_splitter<mask>::HI),
						 _mm_shuffle_pd(S3, S4, _mask_splitter<mask>::LO));
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
			friend class dvec4;

			public:
				inline operator const dvec4 () const {
					return shuffle<mask>(v);
				}

				inline double operator[](int index) const {
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

				const double &x, &y, &z, &w;
				const double &r, &g, &b, &a;
				const double &s, &t, &p, &q;

			private:
					// This massive constructor maps a vector to references
				inline _swzl_ro(const dvec4 &v):
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
				const dvec4 &v;
		};

			// Swizzle helper (Read/Write)
		template <unsigned mask>
		struct _swzl_rw
		{
			friend class dvec4;

			public:
				inline operator const dvec4 () const {
					return shuffle<mask>(v);
				}

				inline double& operator[](int index) {
					return v[(mask >> (index << 1)) & 0x3];
				}

					// Swizzle from dvec4
				inline dvec4& operator = (const dvec4 &r) {
					return v = shuffle<_mask_reverser<mask>::MASK>(r);
				}

					// Swizzle from same r/o mask (v1.xyzw = v2.xyzw)
				inline dvec4& operator = (const _swzl_ro<mask> &s) {
					return v = s.v;
				}

					// Swizzle from same mask (v1.xyzw = v2.xyzw)
				inline dvec4& operator = (const _swzl_rw &s) {
					return v = s.v;
				}

					// Swizzle mask => other_mask, r/o (v1.zwxy = v2.xyxy)
				template<unsigned other_mask>
				inline dvec4& operator = (const _swzl_ro<other_mask> &s) {
					typedef _mask_merger<other_mask, _mask_reverser<mask>::MASK> merged;

					return v = shuffle<merged::MASK>(s.v);
				}

					// Swizzle mask => other_mask (v1.zwxy = v2.xyzw)
				template<unsigned other_mask>
				inline dvec4& operator = (const _swzl_rw<other_mask> &s) {
					typedef _mask_merger<other_mask, _mask_reverser<mask>::MASK> merged;

					return v = shuffle<merged::MASK>(s.v);
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

				inline dvec4& operator += (double s) {
					return v += s;
				}

				inline dvec4& operator += (const dvec4 &v0) {
					return v += v0.shuffle4_ro4<mask>();
				}

				inline dvec4& operator -= (double s) {
					return v -= s;
				}

				inline dvec4& operator -= (const dvec4 &v0) {
					return v -= v0.shuffle4_ro4<mask>();
				}

				inline dvec4& operator *= (double s) {
					return v *= s;
				}

				inline dvec4& operator *= (const dvec4 &v0) {
					return v *= v0.shuffle4_ro4<mask>();
				}

				inline dvec4& operator /= (double s) {
					return v /= s;
				}

				inline dvec4& operator /= (const dvec4 &v0) {
					return v /= v0.shuffle4_ro4<mask>();
				}

				// ----------------------------------------------------------------- //

				double &x, &y, &z, &w;
				double &r, &g, &b, &a;
				double &s, &t, &p, &q;

			private:
					// This massive contructor maps a vector to references
				inline _swzl_rw(dvec4 &v):
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
				dvec4 &v;
		};

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
/*
		friend inline const dvec4 pow(const dvec4 &v0, const dvec4 &v1) {
			// TODO
		}

		friend inline const dvec4 exp(const dvec4 &v) {
			// TODO
		}
*/
		friend inline const dvec4 log(const dvec4 &v) {
			return log2(v) * 0.69314718055994530942;
		}
/*
		friend inline const dvec4 exp2(const dvec4 &v) {
			// TODO
		}
*/
		friend inline const dvec4 log2(const dvec4 &v) {
			__m128d p0  = _mm_set1_pd( 3.61276447184348752e-05);
			__m128d p1  = _mm_set1_pd(-4.16662127033480827e-04);
			__m128d p2  = _mm_set1_pd(-1.43988260692073185e-01);
			__m128d p3  = _mm_set1_pd( 1.60245637034704267e-01);
			__m128d p4  = _mm_set1_pd( 2.28193656337578229e-03);
			__m128d p5  = _mm_set1_pd(-1.80329036970820794e-01);
			__m128d p6  = _mm_set1_pd(-7.93793829370930689e-03);
			__m128d p7  = _mm_set1_pd( 2.06098446037376922e-01);
			__m128d p8  = _mm_set1_pd( 1.98461565426430164e-02);
			__m128d p9  = _mm_set1_pd(-2.40449108727688962e-01);
			__m128d p10 = _mm_set1_pd(-3.84093543662501949e-02);
			__m128d p11 = _mm_set1_pd( 2.88539004851839364e-01);
			__m128d p12 = _mm_set1_pd( 6.08335872067172597e-02);
			__m128d p13 = _mm_set1_pd(-3.60673760117245982e-01);
			__m128d p14 = _mm_set1_pd(-8.27937055456904317e-02);
			__m128d p15 = _mm_set1_pd( 4.80898346961226595e-01);
			__m128d p16 = _mm_set1_pd( 1.01392360727236079e-01);
			__m128d p17 = _mm_set1_pd(-7.21347520444469934e-01);
			__m128d p18 = _mm_set1_pd(-1.16530490533844182e-01);
			__m128d p19 = _mm_set1_pd( 0.44269504088896339e+00);
			__m128d p20 = _mm_set1_pd( 1.30009193360025350e-01);
			__m128d o = _mm_set1_pd(1.0);
			__m128i t = _mm_set1_epi32(1023);
			__m128d c = _mm_castsi128_pd(_mm_set1_epi64x(0x7FF0000000000000LL));
			__m128d co = _mm_and_pd(c, o);
			__m128d f1 = _mm_sub_pd(_mm_or_pd(_mm_andnot_pd(c, v.m1), co), o);
			__m128d f2 = _mm_sub_pd(_mm_or_pd(_mm_andnot_pd(c, v.m2), co), o);
			__m128i a1 = _mm_sub_epi32(_mm_srli_epi32(_mm_castpd_si128(v.m1), 20), t);
			__m128i a2 = _mm_sub_epi32(_mm_srli_epi32(_mm_castpd_si128(v.m2), 20), t);
			__m128d hi1 = _mm_add_pd(_mm_mul_pd(p0, f1), p1);
			__m128d hi2 = _mm_add_pd(_mm_mul_pd(p0, f2), p1);
			__m128d lo1 = _mm_add_pd(_mm_mul_pd(p2, f1), p3);
			__m128d lo2 = _mm_add_pd(_mm_mul_pd(p2, f2), p3);
			hi1 = _mm_add_pd(_mm_mul_pd(f1, hi1), p4);
			hi2 = _mm_add_pd(_mm_mul_pd(f2, hi2), p4);
			lo1 = _mm_add_pd(_mm_mul_pd(f1, lo1), p5);
			lo2 = _mm_add_pd(_mm_mul_pd(f2, lo2), p5);
			hi1 = _mm_add_pd(_mm_mul_pd(f1, hi1), p6);
			hi2 = _mm_add_pd(_mm_mul_pd(f2, hi2), p6);
			lo1 = _mm_add_pd(_mm_mul_pd(f1, lo1), p7);
			lo2 = _mm_add_pd(_mm_mul_pd(f2, lo2), p7);
			hi1 = _mm_add_pd(_mm_mul_pd(f1, hi1), p8);
			hi2 = _mm_add_pd(_mm_mul_pd(f2, hi2), p8);
			lo1 = _mm_add_pd(_mm_mul_pd(f1, lo1), p9);
			lo2 = _mm_add_pd(_mm_mul_pd(f2, lo2), p9);
			hi1 = _mm_add_pd(_mm_mul_pd(f1, hi1), p10);
			hi2 = _mm_add_pd(_mm_mul_pd(f2, hi2), p10);
			lo1 = _mm_add_pd(_mm_mul_pd(f1, lo1), p11);
			lo2 = _mm_add_pd(_mm_mul_pd(f2, lo2), p11);
			hi1 = _mm_add_pd(_mm_mul_pd(f1, hi1), p12);
			hi2 = _mm_add_pd(_mm_mul_pd(f2, hi2), p12);
			lo1 = _mm_add_pd(_mm_mul_pd(f1, lo1), p13);
			lo2 = _mm_add_pd(_mm_mul_pd(f2, lo2), p13);
			hi1 = _mm_add_pd(_mm_mul_pd(f1, hi1), p14);
			hi2 = _mm_add_pd(_mm_mul_pd(f2, hi2), p14);
			lo1 = _mm_add_pd(_mm_mul_pd(f1, lo1), p15);
			lo2 = _mm_add_pd(_mm_mul_pd(f2, lo2), p15);
			hi1 = _mm_add_pd(_mm_mul_pd(f1, hi1), p16);
			hi2 = _mm_add_pd(_mm_mul_pd(f2, hi2), p16);
			lo1 = _mm_add_pd(_mm_mul_pd(f1, lo1), p17);
			lo2 = _mm_add_pd(_mm_mul_pd(f2, lo2), p17);
			hi1 = _mm_add_pd(_mm_mul_pd(f1, hi1), p18);
			hi2 = _mm_add_pd(_mm_mul_pd(f2, hi2), p18);
			lo1 = _mm_add_pd(_mm_mul_pd(f1, lo1), p19);
			lo2 = _mm_add_pd(_mm_mul_pd(f2, lo2), p19);
			hi1 = _mm_add_pd(_mm_mul_pd(f1, hi1), p20);
			hi2 = _mm_add_pd(_mm_mul_pd(f2, hi2), p20);
			__m128d x21  = _mm_mul_pd(f1, f1);
			__m128d x101 = _mm_mul_pd(x21, x21);
			__m128d x22  = _mm_mul_pd(f2, f2);
			__m128d x102 = _mm_mul_pd(x22, x22);
			x101 = _mm_mul_pd(x101, x101);
			x101 = _mm_mul_pd(x21, x101);
			x102 = _mm_mul_pd(x102, x102);
			x102 = _mm_mul_pd(x22, x102);
			return dvec4(_mm_add_pd(_mm_add_pd(_mm_mul_pd(
								    _mm_add_pd(_mm_mul_pd(x101, hi1), lo1), f1), f1),
											   _mm_cvtepi32_pd(_mm_shuffle_epi32(a1, 0x0D))),
						 _mm_add_pd(_mm_add_pd(_mm_mul_pd(
									_mm_add_pd(_mm_mul_pd(x102, hi2), lo2), f2), f2),
											   _mm_cvtepi32_pd(_mm_shuffle_epi32(a2, 0x0D))));
		}

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
			__m128d h = _mm_set1_pd(0.5);
			return dvec4(_mm_cvtepi32_pd(_mm_cvtpd_epi32(_mm_add_pd(v.m1, h))),
						 _mm_cvtepi32_pd(_mm_cvtpd_epi32(_mm_add_pd(v.m2, h))));
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
			__m128d h = _mm_set1_pd(0.5);
			return dvec4(_mm_cvtepi32_pd(_mm_srai_epi32(
						 _mm_cvtpd_epi32(_mm_sub_pd(_mm_add_pd(v.m1, v.m1), h)), 1)),
						 _mm_cvtepi32_pd(_mm_srai_epi32(
						 _mm_cvtpd_epi32(_mm_sub_pd(_mm_add_pd(v.m2, v.m2), h)), 1)));
		}

		friend inline const dvec4 fract(const dvec4 &v) {
			__m128d h = _mm_set1_pd(0.5);
			return dvec4(_mm_sub_pd(v.m1, _mm_cvtepi32_pd(_mm_srai_epi32(
						 _mm_cvtpd_epi32(_mm_sub_pd(_mm_add_pd(v.m1, v.m1), h)), 1))),
						 _mm_sub_pd(v.m2, _mm_cvtepi32_pd(_mm_srai_epi32(
						 _mm_cvtpd_epi32(_mm_sub_pd(_mm_add_pd(v.m2, v.m2), h)), 1))));
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

		friend inline const dvec4 mod(const dvec4 &v, double d) {
			__m128d h = _mm_set1_pd(0.5);
			__m128d dd = _mm_set1_pd(d);
			__m128d d1 = _mm_div_pd(v.m1, dd);
			__m128d d2 = _mm_div_pd(v.m2, dd);
			return dvec4(_mm_sub_pd(v.m1, _mm_mul_pd(dd, _mm_cvtepi32_pd(
										   _mm_srai_epi32(_mm_cvtpd_epi32(_mm_sub_pd(
										   _mm_add_pd(d1, d1), h)), 1)))),
						 _mm_sub_pd(v.m2, _mm_mul_pd(dd, _mm_cvtepi32_pd(
										   _mm_srai_epi32(_mm_cvtpd_epi32(_mm_sub_pd(
										   _mm_add_pd(d2, d2), h)), 1)))));
		}

		friend inline const dvec4 mod(const dvec4 &v0, const dvec4 &v1) {
			__m128d h = _mm_set1_pd(0.5);
			__m128d d1 = _mm_div_pd(v0.m1, v1.m1);
			__m128d d2 = _mm_div_pd(v0.m2, v1.m2);
			return dvec4(_mm_sub_pd(v0.m1, _mm_mul_pd(v1.m1, _mm_cvtepi32_pd(
										   _mm_srai_epi32(_mm_cvtpd_epi32(_mm_sub_pd(
										   _mm_add_pd(d1, d1), h)), 1)))),
						 _mm_sub_pd(v0.m2, _mm_mul_pd(v1.m2, _mm_cvtepi32_pd(
										   _mm_srai_epi32(_mm_cvtpd_epi32(_mm_sub_pd(
										   _mm_add_pd(d2, d2), h)), 1)))));
		}

		friend inline const dvec4 modf(const dvec4 &v0, dvec4 &v1) {
			__m128d nz = _mm_set1_pd(-0.0);
			v1.m1 = _mm_or_pd(_mm_cvtepi32_pd(_mm_cvttpd_epi32(v0.m1)),
							  _mm_and_pd(nz, v0.m1));
			v1.m2 = _mm_or_pd(_mm_cvtepi32_pd(_mm_cvttpd_epi32(v0.m2)),
							  _mm_and_pd(nz, v0.m2));
			return dvec4(_mm_sub_pd(v0.m1, v1.m1),
						 _mm_sub_pd(v0.m2, v1.m2));
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

		friend inline const dvec4 trunc(const dvec4 &v) {
			__m128d h = _mm_set1_pd(0.5);
			__m128d nz = _mm_set1_pd(-0.0);
			return dvec4(_mm_cvtepi32_pd(_mm_cvtpd_epi32(_mm_sub_pd(v.m1,
										 _mm_or_pd(_mm_and_pd(v.m1, nz),  h)))),
						 _mm_cvtepi32_pd(_mm_cvtpd_epi32(_mm_sub_pd(v.m2,
										 _mm_or_pd(_mm_and_pd(v.m2, nz),  h)))));
		}

		// ----------------------------------------------------------------- //

		friend inline double distance(const dvec4 &v0, const dvec4 &v1) {
			__m128d dd1 = _mm_sub_pd(v0.m1, v1.m1);
			__m128d dd2 = _mm_sub_pd(v0.m2, v1.m2);
			__m128d ll1 = _mm_mul_pd(dd1, dd1);
			__m128d ll2 = _mm_mul_pd(dd2, dd2);
			return _mm_cvtsd_f64(_mm_sqrt_pd(_mm_add_pd(
				_mm_add_pd(ll1, ll2),
				_mm_add_pd(_mm_shuffle_pd(ll1, ll1, 0x01),
						   _mm_shuffle_pd(ll2, ll2, 0x01)))));
		}

		friend inline double dot(const dvec4 &v0, const dvec4 &v1) {
			__m128d ll1 = _mm_mul_pd(v0.m1, v1.m1);
			__m128d ll2 = _mm_mul_pd(v0.m2, v1.m2);
			return _mm_cvtsd_f64(_mm_add_pd(
				_mm_add_pd(ll1, ll2),
				_mm_add_pd(_mm_shuffle_pd(ll1, ll1, 0x01),
						   _mm_shuffle_pd(ll2, ll2, 0x01))));
		}

		friend inline const dvec4 faceforward(const dvec4 &v0,
											  const dvec4 &v1, const dvec4 &v2) {
			__m128d z = _mm_setzero_pd();
			__m128d nz = _mm_set1_pd(-0.0);
			__m128d ll1 = _mm_mul_pd(v2.m1, v1.m1);
			__m128d ll2 = _mm_mul_pd(v2.m2, v1.m2);
			__m128d dot = _mm_add_pd(
						  _mm_add_pd(ll1, ll2),
						  _mm_add_pd(_mm_shuffle_pd(ll1, ll1, 0x01),
									 _mm_shuffle_pd(ll2, ll2, 0x01)));
			return dvec4(_mm_xor_pd(_mm_and_pd(_mm_cmpnlt_pd(dot, z), nz), v0.m1),
						 _mm_xor_pd(_mm_and_pd(_mm_cmpnlt_pd(dot, z), nz), v0.m2));
		}

		friend inline double length(const dvec4 &v) {
			__m128d ll1 = _mm_mul_pd(v.m1, v.m1);
			__m128d ll2 = _mm_mul_pd(v.m2, v.m2);
			return _mm_cvtsd_f64(_mm_sqrt_pd(_mm_add_pd(
				_mm_add_pd(ll1, ll2),
				_mm_add_pd(_mm_shuffle_pd(ll1, ll1, 0x01),
						   _mm_shuffle_pd(ll2, ll2, 0x01)))));
		}

		friend inline const dvec4 normalize(const dvec4 &v) {
			__m128d ll1 = _mm_mul_pd(v.m1, v.m1);
			__m128d ll2 = _mm_mul_pd(v.m2, v.m2);
			__m128d len = _mm_sqrt_pd(_mm_add_pd(
									  _mm_add_pd(ll1, ll2),
									  _mm_add_pd(_mm_shuffle_pd(ll1, ll1, 0x01),
												 _mm_shuffle_pd(ll2, ll2, 0x01))));
			return dvec4(_mm_div_pd(v.m1, len), _mm_div_pd(v.m2, len));
		}

		friend inline const dvec4 reflect(const dvec4 &v0, const dvec4 &v1) {
			__m128d ll1 = _mm_mul_pd(v0.m1, v1.m1);
			__m128d ll2 = _mm_mul_pd(v0.m2, v1.m2);
			__m128d res = _mm_add_pd(_mm_add_pd(ll1, ll2),
									 _mm_add_pd(_mm_shuffle_pd(ll1, ll1, 0x01),
												_mm_shuffle_pd(ll2, ll2, 0x01)));
			res = _mm_add_pd(res, res);
			return dvec4(_mm_sub_pd(v0.m1, _mm_mul_pd(res, v1.m1)),
						 _mm_sub_pd(v0.m2, _mm_mul_pd(res, v1.m2)));
		}

		friend inline const dvec4 refract(const dvec4 &v0, const dvec4 &v1,
										  double d) {
			__m128d o = _mm_set1_pd(1.0);
			__m128d z = _mm_set1_pd(0.0);
			__m128d e = _mm_set1_pd(d);
			__m128d ll1 = _mm_mul_pd(v1.m1, v0.m1);
			__m128d ll2 = _mm_mul_pd(v1.m2, v0.m2);
			__m128d dot = _mm_add_pd(_mm_add_pd(ll1, ll2),
									 _mm_add_pd(_mm_shuffle_pd(ll1, ll1, 0x01),
												_mm_shuffle_pd(ll2, ll2, 0x01)));
			__m128d k = _mm_sub_pd(o, _mm_mul_pd(_mm_mul_pd(e, e), _mm_sub_pd(o,
												 _mm_mul_pd(dot, dot))));
			return dvec4(_mm_and_pd(_mm_cmpnlt_pd(k, z), _mm_mul_pd(
									_mm_mul_pd(e, _mm_sub_pd(v0.m1, _mm_mul_pd(
									_mm_mul_pd(e, dot), _mm_sqrt_pd(k)))), v1.m1)),
						 _mm_and_pd(_mm_cmpnlt_pd(k, z), _mm_mul_pd(
									_mm_mul_pd(e, _mm_sub_pd(v0.m2, _mm_mul_pd(
									_mm_mul_pd(e, dot), _mm_sqrt_pd(k)))), v1.m2)));
		}

		// ----------------------------------------------------------------- //

		friend inline bool operator == (const dvec4 &v0, const dvec4 &v1) {
			return _mm_movemask_pd(_mm_and_pd(
								   _mm_cmpeq_pd(v0.m1, v1.m1),
								   _mm_cmpeq_pd(v0.m2, v1.m2))) == 0x03;
		}

		friend inline bool operator != (const dvec4 &v0, const dvec4 &v1) {
			return _mm_movemask_pd(_mm_and_pd(
								   _mm_cmpneq_pd(v0.m1, v1.m1),
								   _mm_cmpneq_pd(v0.m2, v1.m2))) != 0x00;
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
			__m128d m[2];
			struct {
				__m128d	m1;
				__m128d	m2;
			};
		};
};

#include "swizzle4.h"

// Template specialization for mask 0xE4 (No shuffle)
template <>
inline dvec4 dvec4::shuffle<0xE4>(const dvec4 &v)
{
	return v;
}

#endif
